# bots/london_break_user_bot.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List

from bots.base.tracy import Tracy
from services.mt5.position import Position
from services.mt5.data_fetcher import DataFetcher
from services.mt5.orders import OrderService


# ==========================================================
# BOX STRUCTURE
#  - high/low use candle BODIES (open/close)
#  - size = high - low
# ==========================================================
@dataclass(slots=True)
class Box:
    high: float
    low: float
    size: float

    @property
    def half(self) -> float:
        # 50% level of the box
        return (self.high + self.low) / 2.0


class LondonBreakUserBot(Tracy):
    """
    ✅ London Break (simple, your rules)

    TIME (UTC/GMT):
      - Box window: 22:00 -> 02:00 (4 hours back)
      - Box calculation allowed ONLY between 02:00 and 03:00 UTC
        If we miss it (time >= 03:00), we DO NOT trade today.

    ENTRIES:
      - Breakout trade (2 positions):
          1) TP leg
          2) Runner leg (TP=None, managed by trailing)
      - Reversal trade (2 positions) (YOUR definition):
          If breakout was BUY, reversal is ALSO BUY (same direction)
          Trigger: last CLOSED candle closes below 50% box
          Only take reversal if breakout trades are STILL OPEN.
          If breakout trades closed before reversal triggers => "missed reversal today" once.

    MAGIC:
      4 magics for 4 legs:
        - breakout_tp, breakout_runner
        - reversal_tp, reversal_runner

    NOTIFICATIONS:
      - notifier is passed in by Engine as self.notifier
      - we send to a specific channel by parameter (e.g. "discord" or "email")
      - flags ensure we send/log once

    TRAILING (Runner legs only):
      - Uses Position.refresh_from_mt5_position() and Position.modify_sl_tp()
      - Never calls OrderService directly for modifications (your rule)
      - Trailing rule (simple):
          If distance(price_current, sl) >= 1 * box.size:
              move SL to price_current - 0.5*box.size for BUY
              move SL to price_current + 0.5*box.size for SELL
      - ✅ runs every bot cycle (no throttle, no config)
    """

    def __init__(self, **kwargs):
        super().__init__(
            loop_interval=60,
            enable_friday_preclose=True,
            **kwargs,
        )

        # -----------------------------
        # Tools
        # -----------------------------
        # Share MT5Service's process-wide lock across data and order helpers.
        self._mt5_lock = getattr(self.mt5, "mt5_lock", None) or getattr(self.mt5, "_lock", None)
        self.df = DataFetcher(logger=self.logger, lock=self._mt5_lock) if self._mt5_lock is not None else DataFetcher(logger=self.logger)
        # We still inject OrderService into Position objects for OPEN/CLOSE/MODIFY,
        # but the BOT never calls OrderService directly for modifications.
        self.orders = OrderService(logger=self.logger, lock=self._mt5_lock) if self._mt5_lock is not None else OrderService(logger=self.logger)

        # -----------------------------
        # Config values (with safe defaults)
        # -----------------------------
        if "timeframe" not in self.bot_params:
            raise ValueError("Bot timeframe is missing (BOT_PARAMS_JSON timeframe)")

        self.timeframe = int(self.bot_params["timeframe"])

        if "lot" not in self.bot_params:
            raise ValueError("Bot lot is missing (BOT_PARAMS_JSON lot)")

        self.volume = float(self.bot_params["lot"])

        if "deviation" not in self.bot_params:
            raise ValueError("Bot deviation is missing (BOT_PARAMS_JSON deviation)")

        self.deviation = int(self.bot_params["deviation"])
        # Optional spread gate. 0 disables the filter until you choose a live threshold.
        self.max_spread_points = float(self.bot_params.get("max_spread_points", 0) or 0)
        self._last_spread_block_log = 0.0

        # Box window 22:00 -> 02:00
        self.range_end_hour = 2
        self.range_hours_back = 4

        # Allowed calc window 02:00 -> 03:00 only
        self.calc_start_hour = 2
        self.calc_end_hour = 3 # exclusive

        # Where to send notifications (you can set this per bot)
        # Examples: "discord" or "email"
        ch = getattr(getattr(self.config, "notify", None), "channel", None)
        if not ch:
            raise ValueError("Notification channel is missing (config.notify.channel)")
        self.notify_channel = str(ch).strip().lower()
        self.notify_channels = [x.strip().lower() for x in self.notify_channel.split(",") if x.strip()]

        # -----------------------------
        # Magic numbers (4 legs)
        # -----------------------------
        base_magic = int(getattr(self, "magic", 0) or 0)
        self.magic_break_tp = self.magic_for(0)
        self.magic_break_runner = self.magic_for(1)
        self.magic_rev_tp = self.magic_for(2)
        self.magic_rev_runner = self.magic_for(3)

        # -----------------------------
        # Daily state
        # -----------------------------
        self.box: Optional[Box] = None
        self.day_key: Optional[str] = None

        self.break_side01: Optional[int] = None  # 0=BUY, 1=SELL
        self.break_executed: bool = False
        self.rev_executed: bool = False
        self.rev_missed_today: bool = False

        # Keep tickets so we can check if breakout trades still open
        self.break_tickets: List[int] = []
        self.rev_tickets: List[int] = []

        # If we miss calc window for a day, we lock the day
        self.cant_trade_day_key: Optional[str] = None
        self._state_recovered_day_key: Optional[str] = None

        # -----------------------------
        # Print-once flags (LOG)
        # -----------------------------
        self.msg_wait_box: bool = False
        self.msg_box_done: bool = False
        self.msg_wait_break: bool = False
        self.msg_wait_rev: bool = False
        self.msg_missed_rev: bool = False
        self.msg_cant_trade_today: bool = False

        # -----------------------------
        # Notify-once flags (NOTIFIER)
        # -----------------------------
        self.ntf_wait_box: bool = False
        self.ntf_box_done: bool = False
        self.ntf_wait_break: bool = False
        self.ntf_breakout: bool = False
        self.ntf_wait_rev: bool = False
        self.ntf_missed_rev: bool = False
        self.ntf_rev_trigger: bool = False
        self.ntf_cant_trade: bool = False
        self.ntf_startup: bool = False

        self.logger.info(
            f"[{self.__class__.__name__}] init symbol={self.symbol} base_magic={base_magic} "
            f"leg_magics={self.owned_magics()} "
            f"box=22->02 UTC calc_window=02->03 channels={self.notify_channels}"
        )

    def owned_magic_slots(self) -> List[int]:
        return [0, 1, 2, 3]

    # ==========================================================
    # TIME HELPERS
    # ==========================================================
    def _now(self) -> datetime:
        return datetime.utcnow()

    def _end_dt_02(self, now: datetime) -> datetime:
        # 02:00 for the current day
        return datetime(now.year, now.month, now.day, self.range_end_hour, 0, 0)

    def _mk_day_key(self, end_dt: datetime) -> str:
        return end_dt.strftime("%Y-%m-%d")

    def _in_calc_window(self, now: datetime) -> bool:
        return self.calc_start_hour <= now.hour < self.calc_end_hour

    # ==========================================================
    # NOTIFY (direct, no wrapper logic)
    # ==========================================================
    def _send(
        self,
        message: str,
        meta: Optional[Dict[str, Any]] = None,
        subject: Optional[str] = None,
    ) -> None:
        """
        Direct send using NotificationHub passed as self.notifier.
        No wrappers / no "smart routing". Caller controls flags.
        """
        try:
            if getattr(self, "notifier", None) is None:
                return
            if subject is None:
                # Email subjects come from the event message unless the caller supplies a tighter one.
                subject = str(message).splitlines()[0][:120]
            for channel in self.notify_channels:
                result = self.notifier.send(message, channel=channel, meta=(meta or {}), subject=subject)
                if isinstance(result, dict) and not result.get("ok"):
                    self.logger.warning(f"[{self.symbol}] notification failed channel={channel}: {result}")
        except Exception as e:
            try:
                self.logger.error(f"[{self.symbol}] notifier send failed: {e}")
            except Exception:
                pass

    def _startup_notify_if_possible(self) -> None:
        if self.ntf_startup:
            return
        if getattr(self, "notifier", None) is None:
            return
        self._send(
            f"🤖 {self.bot_name} started",
            subject=f"{self.bot_name} started",
            meta={
                "bot": self.bot_name,
                "market": self.market,
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "volume": self.volume,
                "deviation": self.deviation,
                "notify_channels": self.notify_channels,
                "calc_window_utc": "02:00-03:00",
                "box_window_utc": "22:00-02:00",
                "magic_break_tp": self.magic_break_tp,
                "magic_break_runner": self.magic_break_runner,
                "magic_rev_tp": self.magic_rev_tp,
                "magic_rev_runner": self.magic_rev_runner,
            },
        )
        self.ntf_startup = True

    # ==========================================================
    # PRICE HELPERS (NO MID)
    # ==========================================================
    def _tick(self) -> Optional[Dict[str, float]]:
        """
        Get tick prices (bid/ask).
        BUY uses ASK, SELL uses BID.
        """
        t = self.df.tick(self.symbol)
        if not t:
            return None
        bid = float(t["bid"])
        ask = float(t["ask"])
        return {"bid": bid, "ask": ask, "spread": ask - bid}

    def _symbol_point(self) -> Optional[float]:
        """
        MT5 point size for converting raw spread into broker points.
        """
        try:
            info = self.mt5.symbol_info(self.symbol)
        except Exception as e:
            self.logger.warning(f"[{self.symbol}] spread check skipped: symbol_info failed: {e}")
            return None

        point = getattr(info, "point", None) if info is not None else None
        try:
            point = float(point)
        except (TypeError, ValueError):
            point = 0.0

        if point <= 0:
            self.logger.warning(f"[{self.symbol}] spread check skipped: invalid symbol point={point}")
            return None

        return point

    def _spread_points(self, tick: Dict[str, float]) -> Optional[float]:
        point = self._symbol_point()
        if point is None:
            return None
        return float(tick["spread"]) / float(point)

    def _spread_ok(self, tick: Dict[str, float], *, context: str) -> bool:
        """
        Block entries only when max_spread_points > 0 and the live spread is wider.
        """
        if self.max_spread_points <= 0:
            return True

        spread_points = self._spread_points(tick)
        if spread_points is None:
            return False

        self.logger.info(
            f"[{self.symbol}] {context} spread_points={spread_points:.1f} "
            f"max_spread_points={self.max_spread_points:.1f}"
        )

        if spread_points <= self.max_spread_points:
            return True

        now_ts = time.time()
        if now_ts - self._last_spread_block_log >= 60:
            self.logger.warning(
                f"[{self.symbol}] {context} blocked by spread: "
                f"spread_points={spread_points:.1f} max_spread_points={self.max_spread_points:.1f}"
            )
            self._last_spread_block_log = now_ts

        return False

    def _last_closed_close(self) -> Optional[float]:
        """
        Last CLOSED candle close.
        iloc[-2] because iloc[-1] may still be forming.
        """
        rates = self.df.rates_from_pos(self.symbol, self.timeframe, start_pos=0, count=3, as_df=True)
        if rates is None or getattr(rates, "empty", False):
            return None
        try:
            return float(rates["close"].iloc[-2])
        except Exception:
            return None

    # ==========================================================
    # DAILY RESET HELPERS
    # ==========================================================
    def _reset_daily_flow_flags(self) -> None:
        # LOG flags
        self.msg_wait_box = False
        self.msg_box_done = False
        self.msg_wait_break = False
        self.msg_wait_rev = False
        self.msg_missed_rev = False
        self.msg_cant_trade_today = False

        # NOTIFY flags
        self.ntf_wait_box = False
        self.ntf_box_done = False
        self.ntf_wait_break = False
        self.ntf_breakout = False
        self.ntf_wait_rev = False
        self.ntf_missed_rev = False
        self.ntf_rev_trigger = False
        self.ntf_cant_trade = False

    def _reset_for_new_day(self, *, k: str) -> None:
        self.box = None
        self.day_key = k

        self.break_side01 = None
        self.break_executed = False
        self.rev_executed = False
        self.rev_missed_today = False

        self.break_tickets = []
        self.rev_tickets = []

        # allow trading again unless we miss window later
        self.cant_trade_day_key = None

        self._reset_daily_flow_flags()

    # ==========================================================
    # BOX CALCULATION (YOUR TIME RULES)
    # ==========================================================
    def _calculate_box_if_needed(self, now: datetime) -> None:
        """
        Rules:
          - Only calculate box if we haven't calculated it already for today
          - Only calculate between 02:00 and 03:00 UTC
          - If time >= 03:00 and box not calculated => can't trade today (print + countdown once)
        """
        end_dt = self._end_dt_02(now)
        k = self._mk_day_key(end_dt)

        # If day changed (new k), reset day state
        if self.day_key != k:
            self._reset_for_new_day(k=k)

        # If already calculated box today, stop
        if self.box is not None:
            return

        # If we already marked "can't trade" for this day, do nothing
        if self.cant_trade_day_key == k:
            return

        # Before calc window -> wait (once)
        if now.hour < self.calc_start_hour:
            if not self.msg_wait_box:
                self.logger.info(f"[{self.symbol}] waiting to calculate box (02:00-03:00 UTC)")
                self.msg_wait_box = True

            if not self.ntf_wait_box:
                self._send(
                    f"🕒 {self.symbol}: waiting to calculate London box (02:00-03:00 UTC)",
                    meta={"symbol": self.symbol, "bot": self.bot_name, "day": k},
                    subject=f"{self.symbol} waiting for London box",
                )
                self.ntf_wait_box = True
            return

        # After calc window -> can't trade today (once)
        if now.hour >= self.calc_end_hour:
            self.cant_trade_day_key = k

            next_calc = datetime(now.year, now.month, now.day, self.calc_start_hour, 0, 0) + timedelta(days=1)
            delta = next_calc - now
            secs = max(0, int(delta.total_seconds()))
            h = secs // 3600
            m = (secs % 3600) // 60
            s = secs % 60

            if not self.msg_cant_trade_today:
                self.logger.warning(
                    f"[{self.symbol}] can't trade today (missed box calc window). "
                    f"Next trade chance in {h:02d}:{m:02d}:{s:02d} "
                    f"(next window {next_calc.strftime('%Y-%m-%d %H:%M:%S')} UTC)"
                )
                self.msg_cant_trade_today = True

            if not self.ntf_cant_trade:
                self._send(
                    f"⛔ {self.symbol}: can't trade today (missed box calc). Next window in {h:02d}:{m:02d}:{s:02d} UTC",
                    meta={
                        "symbol": self.symbol,
                        "bot": self.bot_name,
                        "day": k,
                        "next_window_utc": next_calc.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    subject=f"{self.symbol} trading skipped today",
                )
                self.ntf_cant_trade = True
            return

        # We are inside calc window (02:00-03:00) -> calculate box
        dt_to = end_dt
        dt_from = end_dt - timedelta(hours=self.range_hours_back)  # 22:00 -> 02:00

        rates = self.df.rates_range(self.symbol, self.timeframe, dt_from, dt_to, as_df=True)
        if rates is None or getattr(rates, "empty", False):
            self.logger.warning(f"[{self.symbol}] box calc failed: no rates in {dt_from}..{dt_to}")
            return

        body_high = float(rates[["open", "close"]].max(axis=1).max())
        body_low = float(rates[["open", "close"]].min(axis=1).min())
        size = body_high - body_low

        if size <= 0:
            self.logger.warning(f"[{self.symbol}] invalid box size={size}")
            return

        self.box = Box(high=body_high, low=body_low, size=size)

        if not self.msg_box_done:
            self.logger.info(
                f"[{self.symbol}] box calculated day={k} "
                f"high={body_high:.5f} low={body_low:.5f} size={size:.5f} "
                f"(range {dt_from}..{dt_to} UTC)"
            )
            self.msg_box_done = True

        if not self.ntf_box_done:
            self._send(
                f"✅ {self.symbol}: box ready (H={body_high:.5f} L={body_low:.5f} S={size:.5f}) day={k}",
                meta={"symbol": self.symbol, "bot": self.bot_name, "day": k},
                subject=f"{self.symbol} London box ready",
            )
            self.ntf_box_done = True

    # ==========================================================
    # OPEN POSITION (USING YOUR Position OBJECT)
    # ==========================================================
    def _comment(self, *, tag: str, leg: str) -> str:
        """
        Broker-safe trade identifier.
        Keep this compact because several MT5 servers reject long comments.
        """
        d = str(self.day_key or "na").replace("-", "")[:8]
        tag_raw = str(tag or "T").upper()
        tag_s = "B" if tag_raw.startswith("BREAK") else "R" if tag_raw.startswith("REV") else tag_raw[:1]
        leg_raw = str(leg or "L").upper()
        leg_s = "R" if leg_raw.startswith("RUN") else "T" if leg_raw.startswith("TP") else leg_raw[:1]
        symbol_s = "".join(ch for ch in str(self.symbol).upper() if ch.isalnum())[:4]
        return f"LB{d}{tag_s}{leg_s}{symbol_s}"[:20]

    def _comment_matches_day(self, comment: str) -> bool:
        if not self.day_key:
            return False

        raw = str(comment or "")
        day_dash = str(self.day_key)
        day_compact = day_dash.replace("-", "")
        compact = "".join(ch for ch in raw.upper() if ch.isalnum())
        return (
            f"LB|DAY={day_dash}|" in raw
            or f"LB_{day_compact}_" in raw
            or compact.startswith(f"LB{day_compact}")
        )

    def _open_position(
        self,
        *,
        side01: int,
        magic: int,
        sl: float,
        tp: Optional[float],
        comment: str,
    ) -> Optional[int]:
        """
        Uses Position.market_order() which handles DB+Redis.
        Returns the discovered MT5 position ticket if successful.
        """
        p = Position(
            logger=self.logger,
            db=self.db,
            cache=self.cache,
            orders=self.orders,
            bot_name=self.bot_name,
            market=self.market,
            symbol=self.symbol,
            side01=int(side01),
            volume=float(self.volume),
            mt5_lock=self._mt5_lock,
            deviation=int(self.deviation),
            magic=int(magic),
            comment=str(comment),
            sl=float(sl),
            tp=(float(tp) if tp is not None else None),
        )

        res = p.market_order()

        if res.get("ok") and p.ticket is not None:
            return int(p.ticket)

        return None

    # ==========================================================
    # BREAKOUT DETECTION (NO MID)
    # ==========================================================
    def _check_breakout(self) -> Optional[int]:
        """
        Returns:
          - 0 for BUY breakout
          - 1 for SELL breakout
          - None if no breakout

        Rules:
          - BUY breakout: ASK > box.high
          - SELL breakout: BID < box.low
        """
        if self.box is None:
            return None

        tick = self._tick()
        if tick is None:
            return None

        if tick["ask"] > self.box.high:
            return 0
        if tick["bid"] < self.box.low:
            return 1

        return None

    # ==========================================================
    # CHECK IF BREAKOUT POSITIONS STILL OPEN
    # ==========================================================
    def _any_break_positions_open(self) -> bool:
        """
        How we check if still open:
          - We loop through stored breakout tickets
          - For each ticket we call Position.refresh_from_mt5_position()
          - If it returns not None => MT5 still has the position open
        """
        if not self.break_tickets:
            return False

        for ticket in self.break_tickets:
            try:
                t = int(ticket)
            except Exception:
                continue

            p = Position(
                logger=self.logger,
                db=self.db,
                cache=self.cache,
                orders=self.orders,
                bot_name=self.bot_name,
                market=self.market,
                symbol=self.symbol,
                side01=0,     # placeholder
                volume=0.0,   # placeholder
                mt5_lock=self._mt5_lock,
            )
            p.ticket = t

            snap = p.refresh_from_mt5_position()
            if snap is not None:
                return True

        return False

    def _recover_open_trade_state(self) -> bool:
        """
        Rebuild memory-only trade state from live MT5 positions after restart.
        This only recovers positions still open in MT5.
        """
        try:
            positions = self.mt5.positions_get(symbol=self.symbol)
        except TypeError:
            positions = self.mt5.positions_get()
        except Exception as e:
            self.logger.warning(f"[{self.symbol}] state recovery skipped: positions_get failed: {e}")
            return False

        if not positions:
            return True

        break_magics = {int(self.magic_break_tp), int(self.magic_break_runner)}
        rev_magics = {int(self.magic_rev_tp), int(self.magic_rev_runner)}

        recovered_break = 0
        recovered_rev = 0

        for p in positions:
            d = p._asdict() if hasattr(p, "_asdict") else {}

            try:
                sym = str(d.get("symbol", "") or "").strip()
                magic = int(d.get("magic", 0) or 0)
                ticket = int(d.get("ticket", 0) or 0)
                side01 = int(d.get("type", 0) or 0)
            except Exception:
                continue

            if sym != str(self.symbol).strip() or ticket <= 0:
                continue

            if magic in break_magics:
                if ticket not in self.break_tickets:
                    self.break_tickets.append(ticket)
                    recovered_break += 1
                self.break_executed = True
                if self.break_side01 is None:
                    self.break_side01 = side01

            elif magic in rev_magics:
                if ticket not in self.rev_tickets:
                    self.rev_tickets.append(ticket)
                    recovered_rev += 1
                self.rev_executed = True

        if recovered_break or recovered_rev:
            self.logger.info(
                f"[{self.symbol}] recovered open trade state "
                f"(break={recovered_break}, reversal={recovered_rev})"
            )
        return True

    def _recover_closed_trade_state_from_db(self) -> bool:
        """
        Rebuild same-day completed state from DB rows after restart.
        This only works for rows with structured LondonBreak comments.
        """
        if not self.day_key:
            return False

        try:
            rows = []
            for magic in self.owned_magics():
                rows.extend(
                    self.db.fetch_positions(
                        limit=100,
                        symbol=self.symbol,
                        magic=int(magic),
                        status="closed",
                    ) or []
                )
        except Exception as e:
            self.logger.warning(f"[{self.symbol}] closed-state recovery skipped: {e}")
            return False

        break_seen = False
        rev_seen = False

        for row in rows:
            comment = str(row.get("comment") or "")
            if not self._comment_matches_day(comment):
                continue

            try:
                magic = int(row.get("magic", 0) or 0)
            except Exception:
                continue

            if magic in (int(self.magic_break_tp), int(self.magic_break_runner)):
                break_seen = True
                if self.break_side01 is None:
                    try:
                        self.break_side01 = int(row.get("type", 0) or 0)
                    except Exception:
                        pass

            elif magic in (int(self.magic_rev_tp), int(self.magic_rev_runner)):
                rev_seen = True

        if break_seen and not self.break_executed:
            self.break_executed = True
            self.rev_missed_today = True
            self.logger.info(
                f"[{self.symbol}] recovered closed breakout for {self.day_key}; "
                f"reversal marked missed"
            )

        if rev_seen and not self.rev_executed:
            self.rev_executed = True
            self.logger.info(f"[{self.symbol}] recovered closed reversal for {self.day_key}")

        return True

    # ==========================================================
    # REVERSAL TRIGGER (YOUR RULE)
    # ==========================================================
    def _reversal_triggered(self) -> bool:
        """
        Your rule:
          - only BUY-day reversal
          - trigger when last closed candle CLOSE < 50% level (box.half)
        """
        if self.box is None or self.break_side01 is None:
            return False

        if int(self.break_side01) != 0:
            return False

        last_close = self._last_closed_close()
        if last_close is None:
            return False

        return last_close < self.box.half

    # ==========================================================
    # TRAILING (Runner legs only)  ✅ CHANGED
    # ==========================================================
    def _trail_runners(self, now: datetime) -> None:
        """
        Simple trailing every bot cycle (no throttle, no config).

        Rule:
          If distance(price_current, sl) >= 1 * box.size:
              move SL to price_current - 0.5*box.size for BUY
              move SL to price_current + 0.5*box.size for SELL

        Trails ONLY runner legs:
          - magic_break_runner
          - magic_rev_runner
        """
        if self.box is None:
            return

        # gather open tickets (prefer Redis open set, else DB)
        tickets: List[int] = []

        # Redis open set (BotNode uses a per-bot symbol key)
        try:
            fn = getattr(self.cache, "get_open_tickets", None)
            if fn is not None:
                raw = fn(symbol=self._pos_open_symbol, limit=5000) or []
                tickets = [int(x) for x in raw if str(x).strip().isdigit()]
        except Exception:
            tickets = []

        # DB fallback
        if not tickets:
            try:
                fn = getattr(self.db, "fetch_open_tickets", None)
                if fn is not None:
                    raw = fn(limit=5000, symbol=str(self.symbol).upper()) or []
                    tickets = [int(x) for x in raw if str(x).strip().isdigit()]
            except Exception:
                tickets = []

        if not tickets:
            return

        trigger_dist = float(self.box.size)
        step = float(self.box.size) * 0.5  # ✅ half the box size

        for ticket in tickets:
            p = Position(
                logger=self.logger,
                db=self.db,
                cache=self.cache,
                orders=self.orders,
                bot_name=self.bot_name,
                market=self.market,
                symbol=self.symbol,
                side01=0,
                volume=0.0,
                mt5_lock=self._mt5_lock,
            )
            p.ticket = int(ticket)

            snap = p.refresh_from_mt5_position()
            if snap is None:
                continue

            raw = snap[1] or {}

            mt5_symbol = str(raw.get("symbol", "") or "").strip()
            if mt5_symbol != str(self.symbol).strip():
                continue

            magic = int(raw.get("magic", 0) or 0)
            if magic not in (int(self.magic_break_runner), int(self.magic_rev_runner)):
                continue

            price_current = raw.get("price_current", None)
            sl = raw.get("sl", None)
            side01 = int(raw.get("type", 0) or 0)

            if price_current is None or sl is None:
                continue

            price_current = float(price_current)
            sl = float(sl)

            if price_current <= 0 or sl <= 0:
                continue

            dist = abs(price_current - sl)
            if dist < trigger_dist:
                continue

            # BUY -> SL below price, SELL -> SL above price
            new_sl = (price_current - step) if side01 == 0 else (price_current + step)

            # never worsen SL
            if side01 == 0 and new_sl <= sl:
                continue
            if side01 == 1 and new_sl >= sl:
                continue

            # keep Position fields consistent (for DB payload)
            p.magic = int(magic)
            p.side01 = int(side01)
            p.sl = float(new_sl)

            res = p.modify_sl_tp(sl=float(new_sl), tp=None)

            if res.get("ok"):
                self.logger.info(
                    f"[{self.symbol}] trail runner ticket={ticket} magic={magic} "
                    f"price_current={price_current:.5f} old_sl={sl:.5f} new_sl={new_sl:.5f} "
                    f"(step=0.5*box={step:.5f})"
                )

    # ==========================================================
    # STRATEGY STEP (standardized)
    # ==========================================================
    def step(self):
        now = self._now()

        self._startup_notify_if_possible()

        # 1) Calculate box (or decide can't trade today)
        self._calculate_box_if_needed(now)

        # If no box, we can't trade (yet or today)
        if self.box is None:
            return

        # If we missed window and locked today, stop
        if self.cant_trade_day_key == self.day_key:
            return

        if self._state_recovered_day_key != self.day_key:
            open_recovered = self._recover_open_trade_state()
            closed_recovered = self._recover_closed_trade_state_from_db()
            if open_recovered and closed_recovered:
                self._state_recovered_day_key = self.day_key

        # ✅ Always trail runners (runs every cycle, but only moves SL when rule is met)
        self._trail_runners(now)

        # 2) Breakout stage
        if not self.break_executed:
            if not self.msg_wait_break:
                self.logger.info(f"[{self.symbol}] box ready → waiting for breakout")
                self.msg_wait_break = True

            if not self.ntf_wait_break:
                self._send(
                    f"📦 {self.symbol}: box ready → waiting for breakout",
                    meta={"symbol": self.symbol, "bot": self.bot_name, "day": self.day_key},
                    subject=f"{self.symbol} waiting for breakout",
                )
                self.ntf_wait_break = True

            side01 = self._check_breakout()
            if side01 is None:
                return

            self.break_side01 = int(side01)

            # SL opposite side of box
            sl = self.box.low if side01 == 0 else self.box.high

            # Current price for TP calc:
            # BUY uses ASK, SELL uses BID
            tick = self._tick()
            if tick is None:
                return
            if not self._spread_ok(tick, context="breakout entry"):
                return

            c_price = tick["ask"] if side01 == 0 else tick["bid"]
            tp = (c_price + self.box.size) if side01 == 0 else (c_price - self.box.size)

            self.logger.info(
                f"[{self.symbol}] breakout detected side={'BUY' if side01==0 else 'SELL'} "
                f"c_price={c_price:.5f} → opening 2 legs"
            )

            if not self.ntf_breakout:
                self._send(
                    f"🚀 {self.symbol}: breakout {('BUY' if side01==0 else 'SELL')} @ {c_price:.5f} → opening 2 legs",
                    meta={
                        "symbol": self.symbol,
                        "bot": self.bot_name,
                        "day": self.day_key,
                        "side": "BUY" if side01 == 0 else "SELL",
                        "c_price": f"{c_price:.5f}",
                        "magic_break_tp": self.magic_break_tp,
                        "magic_break_runner": self.magic_break_runner,
                    },
                    subject=f"{self.symbol} breakout {'BUY' if side01 == 0 else 'SELL'}",
                )
                self.ntf_breakout = True

            # Open TP leg
            t1 = self._open_position(
                side01=side01,
                magic=self.magic_break_tp,
                sl=sl,
                tp=tp,
                comment=self._comment(tag="BREAK", leg="TP"),
            )
            if t1 is not None:
                self.break_tickets.append(int(t1))

            # Open runner leg (no TP)
            t2 = self._open_position(
                side01=side01,
                magic=self.magic_break_runner,
                sl=sl,
                tp=None,
                comment=self._comment(tag="BREAK", leg="RUNNER"),
            )
            if t2 is not None:
                self.break_tickets.append(int(t2))

            if self.break_tickets:
                self.break_executed = True
            else:
                self.logger.warning(
                    f"[{self.symbol}] breakout order attempt failed: no legs opened; will retry"
                )
            return

        # 3) Reversal stage
        if self.break_executed and (not self.rev_executed) and (not self.rev_missed_today):
            if not self.msg_wait_rev:
                self.logger.info(f"[{self.symbol}] waiting for reversal trigger (close < 50% box)")
                self.msg_wait_rev = True

            if not self.ntf_wait_rev:
                self._send(
                    f"🔁 {self.symbol}: waiting for reversal trigger (close < 50% box) — only if break trades still open",
                    meta={"symbol": self.symbol, "bot": self.bot_name, "day": self.day_key},
                    subject=f"{self.symbol} waiting for reversal",
                )
                self.ntf_wait_rev = True

            # reversal only if breakout trades still open
            if not self._any_break_positions_open():
                self.rev_missed_today = True

                if not self.msg_missed_rev:
                    self.logger.warning(f"[{self.symbol}] missed reversal today (break trades not open)")
                    self.msg_missed_rev = True

                if not self.ntf_missed_rev:
                    self._send(
                        f"⚠️ {self.symbol}: missed reversal today (breakout trades already closed)",
                        meta={"symbol": self.symbol, "bot": self.bot_name, "day": self.day_key},
                        subject=f"{self.symbol} reversal missed",
                    )
                    self.ntf_missed_rev = True

                return

            # not triggered yet -> keep waiting
            if not self._reversal_triggered():
                return

            # Your rule: reversal is BUY if breakout day is BUY
            side01 = 0  # BUY
            sl = self.box.low

            tick = self._tick()
            if tick is None:
                return
            if not self._spread_ok(tick, context="reversal entry"):
                return

            c_price = tick["ask"]
            tp = c_price + self.box.size

            self.logger.info(
                f"[{self.symbol}] reversal triggered (close < 50%) c_price={c_price:.5f} "
                f"→ opening 2 reversal legs"
            )

            if not self.ntf_rev_trigger:
                self._send(
                    f"✅ {self.symbol}: reversal triggered (close < 50%) → opening 2 reversal legs (BUY)",
                    meta={
                        "symbol": self.symbol,
                        "bot": self.bot_name,
                        "day": self.day_key,
                        "c_price": f"{c_price:.5f}",
                        "magic_rev_tp": self.magic_rev_tp,
                        "magic_rev_runner": self.magic_rev_runner,
                    },
                    subject=f"{self.symbol} reversal triggered",
                )
                self.ntf_rev_trigger = True

            r1 = self._open_position(
                side01=side01,
                magic=self.magic_rev_tp,
                sl=sl,
                tp=tp,
                comment=self._comment(tag="REV", leg="TP"),
            )
            if r1 is not None:
                self.rev_tickets.append(int(r1))

            r2 = self._open_position(
                side01=side01,
                magic=self.magic_rev_runner,
                sl=sl,
                tp=None,
                comment=self._comment(tag="REV", leg="RUNNER"),
            )
            if r2 is not None:
                self.rev_tickets.append(int(r2))

            if self.rev_tickets:
                self.rev_executed = True
            else:
                self.logger.warning(
                    f"[{self.symbol}] reversal order attempt failed: no legs opened; will retry"
                )
            return

        # 4) Done for now
        return
