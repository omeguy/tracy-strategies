# bots/london_break_flip.py

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from bots.base.tracy import Tracy
from services.mt5.position import Position
from services.mt5.data_fetcher import DataFetcher
from services.mt5.orders import OrderService


@dataclass(slots=True)
class Box:
    high: float
    low: float
    size: float

    @property
    def half(self) -> float:
        return (self.high + self.low) / 2.0


class LondonBreakFlip(Tracy):
    """
    London Break Flip rules:
      1. Calculate box from candle bodies between 22:00 and 02:00 UTC.
      2. Only calculate box between 02:00 and 03:00 UTC.
      3. Breakout enters only after a CLOSED candle closes above high or below low.
      4. Open 2 breakout legs: TP leg + runner leg.
      5. TP leg opens with TP=None first, then TP is set from actual MT5 price_open.
      6. Runner trails only after price moves 1 full box size from actual entry.
      7. If breakout closes by TP/profit, the trading day is finished. No flip.
      8. If breakout closes by SL, wait for CLOSED candle beyond the SL side, then open opposite flip.
      9. Flip behaves like breakout: TP leg + runner leg, TP set after actual fill.
      10. Tickets are stored and recovered so restart does not duplicate same-day trades.
    """

    def __init__(self, **kwargs):
        super().__init__(
            loop_interval=60,
            enable_friday_preclose=True,
            **kwargs,
        )

        self._mt5_lock = getattr(self.mt5, "mt5_lock", None) or getattr(self.mt5, "_lock", None)
        self.df = DataFetcher(logger=self.logger, lock=self._mt5_lock) if self._mt5_lock is not None else DataFetcher(logger=self.logger)
        self.orders = OrderService(logger=self.logger, lock=self._mt5_lock) if self._mt5_lock is not None else OrderService(logger=self.logger)

        if "timeframe" not in self.bot_params:
            raise ValueError("Bot timeframe is missing (BOT_PARAMS_JSON timeframe)")
        if "lot" not in self.bot_params:
            raise ValueError("Bot lot is missing (BOT_PARAMS_JSON lot)")
        if "deviation" not in self.bot_params:
            raise ValueError("Bot deviation is missing (BOT_PARAMS_JSON deviation)")

        self.timeframe = int(self.bot_params["timeframe"])
        self.volume = float(self.bot_params["lot"])
        self.deviation = int(self.bot_params["deviation"])
        self.max_spread_points = float(self.bot_params.get("max_spread_points", 0) or 0)
        self._last_spread_block_log = 0.0

        # Box window: 22:00 -> 02:00, calculation window: 02:00 -> 03:00 UTC.
        self.range_end_hour = 2
        self.range_hours_back = 4
        self.calc_start_hour = 2
        self.calc_end_hour = 3

        ch = getattr(getattr(self.config, "notify", None), "channel", None)
        if not ch:
            raise ValueError("Notification channel is missing (config.notify.channel)")
        self.notify_channel = str(ch).strip().lower()
        self.notify_channels = [x.strip().lower() for x in self.notify_channel.split(",") if x.strip()]

        base_magic = int(getattr(self, "magic", 0) or 0)
        self.magic_break_tp = self.magic_for(0)
        self.magic_break_runner = self.magic_for(1)
        self.magic_rev_tp = self.magic_for(2)       # used as flip TP magic
        self.magic_rev_runner = self.magic_for(3)   # used as flip runner magic

        self.box: Optional[Box] = None
        self.day_key: Optional[str] = None
        self.break_side01: Optional[int] = None  # 0=BUY, 1=SELL
        self.break_executed = False
        self.rev_executed = False                # means flip executed
        self.rev_missed_today = False            # used to block flip/day after TP/profit

        self.break_tickets: List[int] = []
        self.rev_tickets: List[int] = []
        self.break_tp_ticket: Optional[int] = None
        self.break_runner_ticket: Optional[int] = None
        self.flip_tp_ticket: Optional[int] = None
        self.flip_runner_ticket: Optional[int] = None

        self.pending_tp_updates: Dict[int, Dict[str, Any]] = {}
        self.breakout_done_reason: Optional[str] = None  # open/sl/tp/unknown
        self.break_close_reason: Dict[int, str] = {}

        self.cant_trade_day_key: Optional[str] = None
        self._state_recovered_day_key: Optional[str] = None

        self.msg_wait_box = False
        self.msg_box_done = False
        self.msg_wait_break = False
        self.msg_wait_rev = False
        self.msg_missed_rev = False
        self.msg_cant_trade_today = False

        self.ntf_wait_box = False
        self.ntf_box_done = False
        self.ntf_wait_break = False
        self.ntf_breakout = False
        self.ntf_wait_rev = False
        self.ntf_missed_rev = False
        self.ntf_rev_trigger = False
        self.ntf_cant_trade = False
        self.ntf_startup = False

        self.logger.info(
            f"[{self.__class__.__name__}] init symbol={self.symbol} base_magic={base_magic} "
            f"leg_magics={self.owned_magics()} box=22->02 UTC calc_window=02->03 "
            f"channels={self.notify_channels}"
        )

    def owned_magic_slots(self) -> List[int]:
        return [0, 1, 2, 3]

    def _now(self) -> datetime:
        return datetime.utcnow()

    def _end_dt_02(self, now: datetime) -> datetime:
        return datetime(now.year, now.month, now.day, self.range_end_hour, 0, 0)

    def _mk_day_key(self, end_dt: datetime) -> str:
        return end_dt.strftime("%Y-%m-%d")

    def _send(self, message: str, meta: Optional[Dict[str, Any]] = None, subject: Optional[str] = None) -> None:
        try:
            if getattr(self, "notifier", None) is None:
                return
            if subject is None:
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
        if self.ntf_startup or getattr(self, "notifier", None) is None:
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
                "calc_window_utc": "02:00-03:00",
                "box_window_utc": "22:00-02:00",
                "magic_break_tp": self.magic_break_tp,
                "magic_break_runner": self.magic_break_runner,
                "magic_flip_tp": self.magic_rev_tp,
                "magic_flip_runner": self.magic_rev_runner,
            },
        )
        self.ntf_startup = True

    def _tick(self) -> Optional[Dict[str, float]]:
        t = self.df.tick(self.symbol)
        if not t:
            return None
        bid = float(t["bid"])
        ask = float(t["ask"])
        return {"bid": bid, "ask": ask, "spread": ask - bid}

    def _symbol_point(self) -> Optional[float]:
        try:
            info = self.mt5.symbol_info(self.symbol)
        except Exception as e:
            self.logger.warning(f"[{self.symbol}] symbol_info failed: {e}")
            return None
        try:
            point = float(getattr(info, "point", None) or 0.0)
        except Exception:
            point = 0.0
        return point if point > 0 else None

    def _spread_points(self, tick: Dict[str, float]) -> Optional[float]:
        point = self._symbol_point()
        if point is None:
            return None
        return float(tick["spread"]) / point

    def _spread_ok(self, tick: Dict[str, float], *, context: str) -> bool:
        if self.max_spread_points <= 0:
            return True
        spread_points = self._spread_points(tick)
        if spread_points is None:
            return False
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
        rates = self.df.rates_from_pos(self.symbol, self.timeframe, start_pos=0, count=3, as_df=True)
        if rates is None or getattr(rates, "empty", False):
            return None
        try:
            return float(rates["close"].iloc[-2])
        except Exception:
            return None

    def _reset_daily_flow_flags(self) -> None:
        self.msg_wait_box = False
        self.msg_box_done = False
        self.msg_wait_break = False
        self.msg_wait_rev = False
        self.msg_missed_rev = False
        self.msg_cant_trade_today = False
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
        self.break_tp_ticket = None
        self.break_runner_ticket = None
        self.flip_tp_ticket = None
        self.flip_runner_ticket = None
        self.pending_tp_updates = {}
        self.breakout_done_reason = None
        self.break_close_reason = {}
        self.cant_trade_day_key = None
        self._reset_daily_flow_flags()

    def _calculate_box_if_needed(self, now: datetime) -> None:
        end_dt = self._end_dt_02(now)
        k = self._mk_day_key(end_dt)

        if self.day_key != k:
            self._reset_for_new_day(k=k)

        if self.box is not None or self.cant_trade_day_key == k:
            return

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

        if now.hour >= self.calc_end_hour:
            self.cant_trade_day_key = k
            next_calc = datetime(now.year, now.month, now.day, self.calc_start_hour, 0, 0) + timedelta(days=1)
            delta = next_calc - now
            secs = max(0, int(delta.total_seconds()))
            h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
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
                    meta={"symbol": self.symbol, "bot": self.bot_name, "day": k},
                    subject=f"{self.symbol} trading skipped today",
                )
                self.ntf_cant_trade = True
            return

        dt_to = end_dt
        dt_from = end_dt - timedelta(hours=self.range_hours_back)
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
                f"[{self.symbol}] box calculated day={k} high={body_high:.5f} "
                f"low={body_low:.5f} size={size:.5f} (range {dt_from}..{dt_to} UTC)"
            )
            self.msg_box_done = True
        if not self.ntf_box_done:
            self._send(
                f"✅ {self.symbol}: box ready (H={body_high:.5f} L={body_low:.5f} S={size:.5f}) day={k}",
                meta={"symbol": self.symbol, "bot": self.bot_name, "day": k},
                subject=f"{self.symbol} London box ready",
            )
            self.ntf_box_done = True

    def _comment(self, *, tag: str, leg: str) -> str:
        d = str(self.day_key or "na").replace("-", "")[:8]
        tag_s = "B" if str(tag).upper().startswith("BREAK") else "F"
        leg_s = "R" if str(leg).upper().startswith("RUN") else "T"
        symbol_s = "".join(ch for ch in str(self.symbol).upper() if ch.isalnum())[:4]
        return f"LB{d}{tag_s}{leg_s}{symbol_s}"[:20]

    def _comment_matches_day(self, comment: str) -> bool:
        if not self.day_key:
            return False
        day_compact = self.day_key.replace("-", "")
        compact = "".join(ch for ch in str(comment or "").upper() if ch.isalnum())
        return compact.startswith(f"LB{day_compact}")

    def _position_shell(self, *, ticket: int, side01: int = 0, magic: int = 0, sl: Optional[float] = None, tp: Optional[float] = None) -> Position:
        p = Position(
            logger=self.logger,
            db=self.db,
            cache=self.cache,
            orders=self.orders,
            bot_name=self.bot_name,
            market=self.market,
            symbol=self.symbol,
            side01=int(side01),
            volume=0.0,
            mt5_lock=self._mt5_lock,
            magic=int(magic or 0),
            comment="RECOVER",
            sl=sl,
            tp=tp,
        )
        p.ticket = int(ticket)
        return p

    def _open_position(self, *, side01: int, magic: int, sl: float, tp: Optional[float], comment: str) -> Optional[Position]:
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
        if not res.get("ok") or p.ticket is None:
            return None

        if p.price_open is None:
            try:
                p.refresh_from_mt5_position()
            except Exception as e:
                self.logger.warning(f"[{self.symbol}] post-open refresh failed ticket={p.ticket}: {e}")
        return p

    def _actual_entry_price(self, p: Position) -> Optional[float]:
        try:
            if p.price_open is not None and float(p.price_open) > 0:
                return float(p.price_open)
        except Exception:
            pass
        try:
            snap = p.refresh_from_mt5_position()
        except Exception as e:
            self.logger.warning(f"[{self.symbol}] entry refresh failed ticket={getattr(p, 'ticket', None)}: {e}")
            return None
        if snap is None:
            return None
        raw = snap[1] or {}
        try:
            entry = float(raw.get("price_open") or 0.0)
        except Exception:
            return None
        return entry if entry > 0 else None

    def _remember_pending_tp(self, ticket: int, *, side01: int, sl: float, context: str, magic: int) -> None:
        try:
            ticket_i = int(ticket)
        except Exception:
            return
        if ticket_i <= 0:
            return
        self.pending_tp_updates[ticket_i] = {
            "side01": int(side01),
            "sl": float(sl),
            "context": str(context),
            "magic": int(magic or 0),
        }

    def _set_tp_from_actual_entry(self, p: Position, *, side01: int, sl: float, context: str, remember_pending: bool = True) -> bool:
        if self.box is None:
            return False
        ticket_i = int(getattr(p, "ticket", 0) or 0)
        entry = self._actual_entry_price(p)
        if entry is None:
            self.logger.warning(f"[{self.symbol}] {context}: cannot set TP; actual entry missing ticket={ticket_i}")
            if remember_pending and ticket_i > 0:
                self._remember_pending_tp(ticket_i, side01=side01, sl=sl, context=context, magic=getattr(p, "magic", 0))
            return False

        tp = entry + self.box.size if int(side01) == 0 else entry - self.box.size
        p.side01 = int(side01)
        p.sl = float(sl)
        p.tp = float(tp)
        res = p.modify_sl_tp(sl=float(sl), tp=float(tp))
        if not res.get("ok"):
            self.logger.warning(f"[{self.symbol}] {context}: TP modify failed ticket={ticket_i} result={res}")
            if remember_pending and ticket_i > 0:
                self._remember_pending_tp(ticket_i, side01=side01, sl=sl, context=context, magic=getattr(p, "magic", 0))
            return False

        self.pending_tp_updates.pop(ticket_i, None)
        self.logger.info(f"[{self.symbol}] {context}: TP set from actual entry ticket={ticket_i} entry={entry:.5f} tp={tp:.5f}")
        return True

    def _retry_pending_tp_updates(self) -> None:
        if self.box is None or not self.pending_tp_updates:
            return
        for ticket, meta in list(self.pending_tp_updates.items()):
            try:
                ticket_i = int(ticket)
                side01 = int(meta.get("side01", 0))
                sl = float(meta.get("sl", 0.0) or 0.0)
                magic = int(meta.get("magic", 0) or 0)
                context = str(meta.get("context", "pending TP"))
            except Exception:
                self.pending_tp_updates.pop(ticket, None)
                continue

            p = self._position_shell(ticket=ticket_i, side01=side01, magic=magic, sl=sl)
            try:
                snap = p.refresh_from_mt5_position()
            except Exception as e:
                self.logger.warning(f"[{self.symbol}] pending TP check failed ticket={ticket_i}: {e}")
                continue
            if snap is None:
                self.logger.warning(f"[{self.symbol}] pending TP removed because ticket is closed ticket={ticket_i}")
                self.pending_tp_updates.pop(ticket_i, None)
                continue
            self._set_tp_from_actual_entry(p, side01=side01, sl=sl, context=f"retry {context}", remember_pending=True)

    def _remember_leg_ticket(self, *, stage: str, leg: str, ticket: int) -> None:
        try:
            t = int(ticket)
        except Exception:
            return
        if t <= 0:
            return
        stage_s = str(stage).lower().strip()
        leg_s = str(leg).lower().strip()

        if stage_s == "break":
            if t not in self.break_tickets:
                self.break_tickets.append(t)
            if leg_s == "tp":
                self.break_tp_ticket = t
            elif leg_s in ("runner", "run"):
                self.break_runner_ticket = t
        else:
            if t not in self.rev_tickets:
                self.rev_tickets.append(t)
            if leg_s == "tp":
                self.flip_tp_ticket = t
            elif leg_s in ("runner", "run"):
                self.flip_runner_ticket = t

    def _check_breakout(self) -> Optional[int]:
        if self.box is None:
            return None
        last_close = self._last_closed_close()
        if last_close is None:
            return None
        if last_close > self.box.high:
            return 0
        if last_close < self.box.low:
            return 1
        return None

    def _flip_triggered(self) -> bool:
        if self.box is None or self.break_side01 is None:
            return False
        last_close = self._last_closed_close()
        if last_close is None:
            return False
        if int(self.break_side01) == 0:
            return last_close < self.box.low
        if int(self.break_side01) == 1:
            return last_close > self.box.high
        return False

    def _refresh_ticket_open(self, ticket: int, *, magic: int = 0) -> Optional[Dict[str, Any]]:
        p = self._position_shell(ticket=int(ticket), magic=int(magic or 0))
        snap = p.refresh_from_mt5_position()
        if snap is None:
            return None
        return snap[1] or {}

    def _live_positions_by_magics(self, magics: List[int]) -> Optional[List[Dict[str, Any]]]:
        wanted = {int(m) for m in magics if int(m) != 0}
        if not wanted:
            return []
        try:
            try:
                positions = self.mt5.positions_get(symbol=self.symbol)
            except TypeError:
                positions = self.mt5.positions_get()
        except Exception as e:
            self.logger.warning(f"[{self.symbol}] live position scan failed: {e}")
            return None
        if not positions:
            return []

        out: List[Dict[str, Any]] = []
        for pos in positions:
            d = pos._asdict() if hasattr(pos, "_asdict") else {}
            try:
                sym = str(d.get("symbol", "") or "").strip()
                magic = int(d.get("magic", 0) or 0)
                ticket = int(d.get("ticket", 0) or 0)
            except Exception:
                continue
            if ticket > 0 and magic in wanted and sym == str(self.symbol).strip():
                out.append(d)
        return out

    def _any_break_positions_open(self) -> bool:
        break_magics = [int(self.magic_break_tp), int(self.magic_break_runner)]
        known_tickets: List[int] = []
        for t in [self.break_tp_ticket, self.break_runner_ticket, *self.break_tickets]:
            try:
                ti = int(t) if t is not None else 0
            except Exception:
                ti = 0
            if ti > 0 and ti not in known_tickets:
                known_tickets.append(ti)

        for ticket in known_tickets:
            try:
                raw = self._refresh_ticket_open(ticket)
            except Exception as e:
                self.logger.warning(f"[{self.symbol}] breakout open check failed ticket={ticket}; assuming open: {e}")
                return True
            if raw is not None:
                try:
                    magic = int(raw.get("magic", 0) or 0)
                except Exception:
                    magic = 0
                if magic in break_magics:
                    return True

        live = self._live_positions_by_magics(break_magics)
        if live is None:
            return True
        if live:
            for raw in live:
                try:
                    ticket = int(raw.get("ticket", 0) or 0)
                    magic = int(raw.get("magic", 0) or 0)
                except Exception:
                    continue
                if magic == int(self.magic_break_tp):
                    self._remember_leg_ticket(stage="break", leg="tp", ticket=ticket)
                elif magic == int(self.magic_break_runner):
                    self._remember_leg_ticket(stage="break", leg="runner", ticket=ticket)
            return True
        return False

    def _fetch_position_rows(self, *, magics: Optional[List[int]] = None, status: Optional[str] = None, limit: int = 5000) -> List[Dict[str, Any]]:
        try:
            return self.db.fetch_positions(
                limit=limit,
                symbol=self.symbol,
                magics=magics if magics is not None else self.owned_magics(),
                status=status,
            ) or []
        except Exception as e:
            self.logger.warning(f"[{self.symbol}] fetch positions failed: {e}")
            return []

    def _db_row_for_ticket(self, ticket: int) -> Optional[Dict[str, Any]]:
        try:
            ticket_i = int(ticket)
        except Exception:
            return None
        for row in self._fetch_position_rows(limit=5000):
            try:
                if int(row.get("ticket", 0) or 0) == ticket_i:
                    return row
            except Exception:
                continue
        return None

    def _collect_breakout_tickets_from_db(self) -> List[int]:
        if not self.day_key:
            return []
        out: List[int] = []
        rows = self._fetch_position_rows(magics=[int(self.magic_break_tp), int(self.magic_break_runner)], status=None, limit=5000)
        for row in rows:
            if not self._comment_matches_day(str(row.get("comment") or "")):
                continue
            try:
                ticket = int(row.get("ticket", 0) or 0)
                magic = int(row.get("magic", 0) or 0)
            except Exception:
                continue
            if ticket <= 0:
                continue
            if magic == int(self.magic_break_tp):
                self._remember_leg_ticket(stage="break", leg="tp", ticket=ticket)
            elif magic == int(self.magic_break_runner):
                self._remember_leg_ticket(stage="break", leg="runner", ticket=ticket)
            if ticket not in out:
                out.append(ticket)
        return out

    def _as_float_or_none(self, value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            x = float(value)
            return x if x != 0.0 else None
        except Exception:
            return None

    def _price_tolerance(self) -> float:
        point = self._symbol_point()
        if point is not None and point > 0:
            return point * 5.0
        if self.box is not None and self.box.size > 0:
            return self.box.size * 0.02
        return 0.0

    def _history_close_for_ticket(self, ticket: int, *, side01: int = 0, magic: int = 0) -> Optional[Dict[str, Any]]:
        try:
            p = self._position_shell(ticket=int(ticket), side01=int(side01), magic=int(magic))
            hist = p.finalize_from_history()
        except Exception as e:
            self.logger.warning(f"[{self.symbol}] history close lookup failed ticket={ticket}: {e}")
            return None
        if not isinstance(hist, dict) or not hist.get("ok"):
            return None
        return {
            "ticket": int(ticket),
            "status": "closed",
            "close_price": p.close_price,
            "close_time": p.close_time,
            "total_pnl": p.total_pnl,
            "profit": p.profit,
            "history": hist,
        }

    def _classify_breakout_closed_ticket(self, ticket: int, *, magic: int) -> str:
        try:
            ticket_i = int(ticket)
        except Exception:
            return "unknown"
        if ticket_i <= 0:
            return "unknown"

        try:
            raw = self._refresh_ticket_open(ticket_i, magic=int(magic or 0))
        except Exception as e:
            self.logger.warning(f"[{self.symbol}] close classifier failed ticket={ticket_i}; blocking flip: {e}")
            return "unknown"
        if raw is not None:
            return "open"

        row = self._db_row_for_ticket(ticket_i) or {}
        try:
            side01 = int(row.get("type", self.break_side01 if self.break_side01 is not None else 0) or 0)
        except Exception:
            side01 = int(self.break_side01) if self.break_side01 is not None else 0

        sl = self._as_float_or_none(row.get("sl"))
        tp = self._as_float_or_none(row.get("tp"))
        price_open = self._as_float_or_none(row.get("price_open"))
        close_price = self._as_float_or_none(row.get("close_price"))
        total_pnl = self._as_float_or_none(row.get("total_pnl"))
        profit = self._as_float_or_none(row.get("profit"))

        if close_price is None:
            hist = self._history_close_for_ticket(ticket_i, side01=side01, magic=int(magic or 0))
            if hist:
                close_price = self._as_float_or_none(hist.get("close_price"))
                total_pnl = total_pnl if total_pnl is not None else self._as_float_or_none(hist.get("total_pnl"))
                profit = profit if profit is not None else self._as_float_or_none(hist.get("profit"))

        if close_price is None:
            return "unknown"

        tol = self._price_tolerance()

        # If TP leg hit TP, or any breakout leg exits profitably, breakout succeeded.
        if tp is not None and abs(close_price - tp) <= tol:
            return "tp"
        if total_pnl is not None and total_pnl > 0:
            return "tp"
        if profit is not None and profit > 0:
            return "tp"
        if price_open is not None:
            if side01 == 0 and close_price > price_open:
                return "tp"
            if side01 == 1 and close_price < price_open:
                return "tp"

        if sl is not None and abs(close_price - sl) <= tol:
            return "sl"
        if self.box is not None:
            if side01 == 0 and close_price <= self.box.low + tol:
                return "sl"
            if side01 == 1 and close_price >= self.box.high - tol:
                return "sl"

        self.logger.warning(
            f"[{self.symbol}] cannot classify breakout close ticket={ticket_i}: "
            f"close={close_price} sl={sl} tp={tp} entry={price_open} pnl={total_pnl}"
        )
        return "unknown"

    def _breakout_close_status(self) -> str:
        if self._any_break_positions_open():
            self.breakout_done_reason = "open"
            return "open"

        known_tickets: List[int] = []
        for t in [self.break_tp_ticket, self.break_runner_ticket, *self.break_tickets]:
            try:
                ti = int(t) if t is not None else 0
            except Exception:
                ti = 0
            if ti > 0 and ti not in known_tickets:
                known_tickets.append(ti)
        if not known_tickets:
            known_tickets = self._collect_breakout_tickets_from_db()
        if not known_tickets:
            self.breakout_done_reason = "unknown"
            return "unknown"

        results: Dict[int, str] = {}
        for ticket in known_tickets:
            magic = 0
            if int(ticket) == int(self.break_tp_ticket or 0):
                magic = int(self.magic_break_tp)
            elif int(ticket) == int(self.break_runner_ticket or 0):
                magic = int(self.magic_break_runner)
            else:
                row = self._db_row_for_ticket(ticket) or {}
                try:
                    magic = int(row.get("magic", 0) or 0)
                except Exception:
                    magic = 0
            reason = self._classify_breakout_closed_ticket(ticket, magic=magic)
            results[int(ticket)] = reason
            self.break_close_reason[int(ticket)] = reason

        if any(r == "open" for r in results.values()):
            self.breakout_done_reason = "open"
            return "open"
        if any(r == "tp" for r in results.values()):
            self.breakout_done_reason = "tp"
            self.logger.info(f"[{self.symbol}] breakout completed successfully; flip blocked today reasons={results}")
            return "tp"
        if any(r == "unknown" for r in results.values()):
            self.breakout_done_reason = "unknown"
            self.logger.warning(f"[{self.symbol}] breakout close reason unknown; waiting/retrying reasons={results}")
            return "unknown"
        if results and all(r == "sl" for r in results.values()):
            self.breakout_done_reason = "sl"
            self.logger.info(f"[{self.symbol}] breakout closed by SL; flip can wait for candle confirmation reasons={results}")
            return "sl"
        self.breakout_done_reason = "unknown"
        return "unknown"

    def _recover_open_trade_state(self) -> bool:
        try:
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
        flip_magics = {int(self.magic_rev_tp), int(self.magic_rev_runner)}
        recovered_break = 0
        recovered_flip = 0

        for pos in positions:
            d = pos._asdict() if hasattr(pos, "_asdict") else {}
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
                if magic == int(self.magic_break_tp):
                    self._remember_leg_ticket(stage="break", leg="tp", ticket=ticket)
                elif magic == int(self.magic_break_runner):
                    self._remember_leg_ticket(stage="break", leg="runner", ticket=ticket)
                self.break_executed = True
                self.break_side01 = side01 if self.break_side01 is None else self.break_side01
                recovered_break += 1
            elif magic in flip_magics:
                if magic == int(self.magic_rev_tp):
                    self._remember_leg_ticket(stage="flip", leg="tp", ticket=ticket)
                elif magic == int(self.magic_rev_runner):
                    self._remember_leg_ticket(stage="flip", leg="runner", ticket=ticket)
                self.rev_executed = True
                recovered_flip += 1

        if recovered_break or recovered_flip:
            self.logger.info(f"[{self.symbol}] recovered open state break={recovered_break} flip={recovered_flip}")
        return True

    def _recover_closed_trade_state_from_db(self) -> bool:
        if not self.day_key:
            return False
        rows = self._fetch_position_rows(status="closed", limit=5000)
        break_seen = False
        flip_seen = False

        for row in rows:
            if not self._comment_matches_day(str(row.get("comment") or "")):
                continue
            try:
                magic = int(row.get("magic", 0) or 0)
                ticket = int(row.get("ticket", 0) or 0)
            except Exception:
                continue

            if magic in (int(self.magic_break_tp), int(self.magic_break_runner)):
                break_seen = True
                if ticket > 0:
                    if magic == int(self.magic_break_tp):
                        self._remember_leg_ticket(stage="break", leg="tp", ticket=ticket)
                    else:
                        self._remember_leg_ticket(stage="break", leg="runner", ticket=ticket)
                if self.break_side01 is None:
                    try:
                        self.break_side01 = int(row.get("type", 0) or 0)
                    except Exception:
                        pass
            elif magic in (int(self.magic_rev_tp), int(self.magic_rev_runner)):
                flip_seen = True
                if ticket > 0:
                    if magic == int(self.magic_rev_tp):
                        self._remember_leg_ticket(stage="flip", leg="tp", ticket=ticket)
                    else:
                        self._remember_leg_ticket(stage="flip", leg="runner", ticket=ticket)

        if break_seen and not self.break_executed:
            self.break_executed = True
            self.logger.info(f"[{self.symbol}] recovered same-day closed breakout; no duplicate breakout")
        if flip_seen and not self.rev_executed:
            self.rev_executed = True
            self.logger.info(f"[{self.symbol}] recovered same-day closed flip; no duplicate flip")
        return True

    def _trail_runners(self, now: datetime) -> None:
        if self.box is None:
            return

        tickets: List[int] = []
        try:
            fn = getattr(self.cache, "get_open_tickets", None)
            if fn is not None:
                raw = fn(symbol=self._pos_open_symbol, limit=5000) or []
                tickets = [int(x) for x in raw if str(x).strip().isdigit()]
        except Exception:
            tickets = []

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

        trigger_profit = float(self.box.size)
        step = float(self.box.size) * 0.5

        for ticket in tickets:
            p = self._position_shell(ticket=int(ticket))
            snap = p.refresh_from_mt5_position()
            if snap is None:
                continue
            raw = snap[1] or {}
            if str(raw.get("symbol", "") or "").strip() != str(self.symbol).strip():
                continue
            magic = int(raw.get("magic", 0) or 0)
            if magic not in (int(self.magic_break_runner), int(self.magic_rev_runner)):
                continue

            try:
                price_current = float(raw.get("price_current"))
                price_open = float(raw.get("price_open"))
                sl = float(raw.get("sl"))
                side01 = int(raw.get("type", 0) or 0)
            except Exception:
                continue
            if price_current <= 0 or price_open <= 0 or sl <= 0:
                continue

            profit_move = price_current - price_open if side01 == 0 else price_open - price_current
            if profit_move < trigger_profit:
                continue

            new_sl = price_current - step if side01 == 0 else price_current + step
            if side01 == 0 and new_sl <= sl:
                continue
            if side01 == 1 and new_sl >= sl:
                continue

            p.magic = magic
            p.side01 = side01
            p.sl = float(new_sl)
            res = p.modify_sl_tp(sl=float(new_sl), tp=None)
            if res.get("ok"):
                self.logger.info(
                    f"[{self.symbol}] trail runner ticket={ticket} magic={magic} "
                    f"entry={price_open:.5f} current={price_current:.5f} old_sl={sl:.5f} new_sl={new_sl:.5f}"
                )

    def step(self):
        now = self._now()
        self._startup_notify_if_possible()
        self._calculate_box_if_needed(now)

        if self.box is None:
            return
        if self.cant_trade_day_key == self.day_key:
            return

        if self._state_recovered_day_key != self.day_key:
            open_recovered = self._recover_open_trade_state()
            closed_recovered = self._recover_closed_trade_state_from_db()
            if open_recovered and closed_recovered:
                self._state_recovered_day_key = self.day_key

        self._retry_pending_tp_updates()
        self._trail_runners(now)

        # 1) Breakout stage: only once per day.
        if not self.break_executed:
            if not self.msg_wait_break:
                self.logger.info(f"[{self.symbol}] box ready -> waiting for closed-candle breakout")
                self.msg_wait_break = True
            if not self.ntf_wait_break:
                self._send(
                    f"{self.symbol}: box ready -> waiting for closed-candle breakout",
                    meta={"symbol": self.symbol, "bot": self.bot_name, "day": self.day_key},
                    subject=f"{self.symbol} waiting for breakout",
                )
                self.ntf_wait_break = True

            side01 = self._check_breakout()
            if side01 is None:
                return
            self.break_side01 = int(side01)
            sl = self.box.low if side01 == 0 else self.box.high

            tick = self._tick()
            if tick is None or not self._spread_ok(tick, context="breakout entry"):
                return
            signal_price = tick["ask"] if side01 == 0 else tick["bid"]
            self.logger.info(
                f"[{self.symbol}] closed-candle breakout {'BUY' if side01 == 0 else 'SELL'} "
                f"signal_price={signal_price:.5f} -> opening 2 legs"
            )

            if not self.ntf_breakout:
                self._send(
                    f"{self.symbol}: closed-candle breakout {'BUY' if side01 == 0 else 'SELL'} @ {signal_price:.5f}",
                    meta={"symbol": self.symbol, "bot": self.bot_name, "day": self.day_key},
                    subject=f"{self.symbol} breakout {'BUY' if side01 == 0 else 'SELL'}",
                )
                self.ntf_breakout = True

            p1 = self._open_position(
                side01=side01,
                magic=self.magic_break_tp,
                sl=sl,
                tp=None,
                comment=self._comment(tag="BREAK", leg="TP"),
            )
            if p1 is not None and p1.ticket is not None:
                self._remember_leg_ticket(stage="break", leg="tp", ticket=int(p1.ticket))
                self._set_tp_from_actual_entry(p1, side01=side01, sl=sl, context="breakout TP leg")

            p2 = self._open_position(
                side01=side01,
                magic=self.magic_break_runner,
                sl=sl,
                tp=None,
                comment=self._comment(tag="BREAK", leg="RUNNER"),
            )
            if p2 is not None and p2.ticket is not None:
                self._remember_leg_ticket(stage="break", leg="runner", ticket=int(p2.ticket))

            if self.break_tickets:
                self.break_executed = True
            else:
                self.logger.warning(f"[{self.symbol}] breakout order attempt failed: no legs opened; will retry")
            return

        # 2) Flip stage: only after breakout closed by SL, never after TP/profit.
        if self.break_executed and not self.rev_executed and not self.rev_missed_today:
            if not self.msg_wait_rev:
                self.logger.info(f"[{self.symbol}] waiting for flip trigger")
                self.msg_wait_rev = True
            if not self.ntf_wait_rev:
                self._send(
                    f"{self.symbol}: waiting for flip trigger beyond failed breakout SL",
                    meta={"symbol": self.symbol, "bot": self.bot_name, "day": self.day_key},
                    subject=f"{self.symbol} waiting for flip",
                )
                self.ntf_wait_rev = True

            close_status = self._breakout_close_status()
            if close_status == "open":
                return
            if close_status == "tp":
                self.rev_missed_today = True
                self.logger.info(f"[{self.symbol}] breakout hit TP/profit; no flip today")
                return
            if close_status != "sl":
                self.logger.warning(f"[{self.symbol}] breakout close status={close_status}; waiting/retrying before flip")
                return

            if not self._flip_triggered():
                return

            # Opposite direction of breakout.
            side01 = 1 if int(self.break_side01) == 0 else 0
            sl = self.box.high if side01 == 1 else self.box.low

            tick = self._tick()
            if tick is None or not self._spread_ok(tick, context="flip entry"):
                return
            signal_price = tick["bid"] if side01 == 1 else tick["ask"]

            self.logger.info(
                f"[{self.symbol}] flip triggered {'SELL' if side01 == 1 else 'BUY'} "
                f"signal_price={signal_price:.5f} sl={sl:.5f} -> opening 2 legs"
            )

            if not self.ntf_rev_trigger:
                self._send(
                    f"{self.symbol}: flip triggered -> opening 2 legs ({'SELL' if side01 == 1 else 'BUY'})",
                    meta={"symbol": self.symbol, "bot": self.bot_name, "day": self.day_key},
                    subject=f"{self.symbol} flip triggered",
                )
                self.ntf_rev_trigger = True

            r1 = self._open_position(
                side01=side01,
                magic=self.magic_rev_tp,
                sl=sl,
                tp=None,
                comment=self._comment(tag="FLIP", leg="TP"),
            )
            if r1 is not None and r1.ticket is not None:
                self._remember_leg_ticket(stage="flip", leg="tp", ticket=int(r1.ticket))
                self._set_tp_from_actual_entry(r1, side01=side01, sl=sl, context="flip TP leg")

            r2 = self._open_position(
                side01=side01,
                magic=self.magic_rev_runner,
                sl=sl,
                tp=None,
                comment=self._comment(tag="FLIP", leg="RUNNER"),
            )
            if r2 is not None and r2.ticket is not None:
                self._remember_leg_ticket(stage="flip", leg="runner", ticket=int(r2.ticket))

            if self.rev_tickets:
                self.rev_executed = True
            else:
                self.logger.warning(f"[{self.symbol}] flip order attempt failed: no legs opened; will retry")
            return

        return