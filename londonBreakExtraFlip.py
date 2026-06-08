# bots/london_break_extra_flip.py

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


class LondonBreakExtraFlip(Tracy):
    """
    London Break Extra Flip rules:
      1. Calculate box from candle bodies between 22:00 and 02:00 UTC.
      2. Only calculate box between 02:00 and 03:00 UTC.
      3. Breakout enters only after a CLOSED candle closes above high or below low.
      4. Open 2 breakout legs: TP leg + runner leg.
      5. TP leg opens with TP=None first, then TP is set from actual MT5 price_open.
         TP distance = absolute distance between actual entry and SL, so TP and SL are 1:1.
      6. Runner uses real entry-to-SL risk for trailing:
         first 1R move sends SL to break-even; after that, every extra 1R from
         the saved previous trigger point moves SL forward by 0.5R.
      7. If breakout closes by TP/profit, wait until BOTH breakout legs are closed,
         then watch for a possible opposite flip only until the London session ends.
      8. Profit-cycle possible flip triggers only on a CLOSED candle beyond the opposite box side.
      9. If the London session ends before that trigger, trading is finished for the day.
      10. If breakout closes by SL, wait for CLOSED candle beyond the SL side, then open opposite flip.
      11. Flip behaves like breakout: TP leg + runner leg, TP set after actual fill.
      12. Tickets are stored and recovered so restart does not duplicate same-day trades.
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

        # Profit-cycle possible flip window.
        # This is hard-coded for this strategy, not read from BOT_PARAMS_JSON.
        # London session in UTC: 07:00 -> 16:00. Nigeria time: 08:00 -> 17:00.
        self.profit_flip_start_hour = 7
        self.profit_flip_end_hour = 16

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
        self.rev_missed_today = False            # used to block flip/day after session expiry/disabled watch
        self.profit_flip_watch = False           # true after profitable breakout cycle closes
        self.profit_flip_expired = False         # true once London session ends without possible flip

        self.break_tickets: List[int] = []
        self.rev_tickets: List[int] = []
        self.break_tp_ticket: Optional[int] = None
        self.break_runner_ticket: Optional[int] = None
        self.flip_tp_ticket: Optional[int] = None
        self.flip_runner_ticket: Optional[int] = None

        self.pending_tp_updates: Dict[int, Dict[str, Any]] = {}
        # Runner trailing state per MT5 ticket.
        # previous_trigger_point is the last price level that caused a runner SL update.
        # First trigger moves SL to entry. Later triggers move SL by 0.5R steps.
        self.runner_trail_state: Dict[int, Dict[str, Any]] = {}
        self.breakout_done_reason: Optional[str] = None  # open/sl/tp/unknown
        self.break_close_reason: Dict[int, str] = {}

        # Notification guards. These prevent repeated Discord/email alerts for the same ticket/day.
        self.closed_position_ntf_sent: List[int] = []
        self.day_summary_sent = False

        self.cant_trade_day_key: Optional[str] = None
        self._state_recovered_day_key: Optional[str] = None

        self.msg_wait_box = False
        self.msg_box_done = False
        self.msg_wait_break = False
        self.msg_wait_rev = False
        self.msg_missed_rev = False
        self.msg_profit_flip_watch = False
        self.msg_profit_flip_wait_session = False
        self.msg_profit_flip_expired = False
        self.msg_cant_trade_today = False

        self.ntf_wait_box = False
        self.ntf_box_done = False
        self.ntf_wait_break = False
        self.ntf_breakout = False
        self.ntf_wait_rev = False
        self.ntf_missed_rev = False
        self.ntf_profit_flip_watch = False
        self.ntf_profit_flip_wait_session = False
        self.ntf_profit_flip_expired = False
        self.ntf_rev_trigger = False
        self.ntf_cant_trade = False
        self.ntf_startup = False

        self.logger.info(
            f"[{self.__class__.__name__}] init symbol={self.symbol} base_magic={base_magic} "
            f"leg_magics={self.owned_magics()} box=22->02 UTC calc_window=02->03 "
            f"profit_flip_window={self.profit_flip_start_hour:02d}:00-{self.profit_flip_end_hour:02d}:00 UTC "
            f"tp_model=entry_to_sl_1to1 channels={self.notify_channels}"
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



    def _fmt_price(self, value: Any, digits: int = 5) -> str:
        try:
            if value is None:
                return "N/A"
            return f"{float(value):.{digits}f}"
        except Exception:
            return "N/A"

    def _fmt_value(self, value: Any) -> str:
        if value is None:
            return "N/A"
        return str(value)

    def _side_name(self, side01: Optional[int]) -> str:
        if side01 is None:
            return "N/A"
        return "BUY" if int(side01) == 0 else "SELL"

    def _event_subject(self, *, status: str, symbol: Optional[str] = None) -> str:
        sym = symbol or self.symbol
        return f"[{self.bot_name}] {sym} — {status}"

    def _event_message(
        self,
        *,
        status: str,
        emoji: str = "📌",
        details: Optional[Dict[str, Any]] = None,
        levels: Optional[Dict[str, Any]] = None,
        tickets: Optional[Dict[str, Any]] = None,
        note: Optional[str] = None,
    ) -> str:
        now = self._now()
        lines: List[str] = [
            f"{emoji} London Break Extra Flip",
            "━━━━━━━━━━━━━━━━━━━━",
            f"Status: {status}",
            "",
            f"Bot: {self.bot_name}",
            f"Symbol: {self.symbol}",
            f"Market: {self.market}",
            f"Day: {self._fmt_value(self.day_key)}",
            f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC",
        ]

        if details:
            lines.append("")
            lines.append("Details:")
            for key, value in details.items():
                lines.append(f"- {key}: {self._fmt_value(value)}")

        if levels:
            lines.append("")
            lines.append("Levels:")
            for key, value in levels.items():
                lines.append(f"- {key}: {self._fmt_price(value)}")

        if tickets:
            lines.append("")
            lines.append("Tickets:")
            for key, value in tickets.items():
                lines.append(f"- {key}: {self._fmt_value(value)}")

        if note:
            lines.append("")
            lines.append("Note:")
            lines.append(str(note))

        return "\n".join(lines)

    def _send_event(
        self,
        *,
        status: str,
        emoji: str = "📌",
        details: Optional[Dict[str, Any]] = None,
        levels: Optional[Dict[str, Any]] = None,
        tickets: Optional[Dict[str, Any]] = None,
        note: Optional[str] = None,
        subject: Optional[str] = None,
    ) -> None:
        message = self._event_message(
            status=status,
            emoji=emoji,
            details=details,
            levels=levels,
            tickets=tickets,
            note=note,
        )
        meta = {
            "bot": self.bot_name,
            "market": self.market,
            "symbol": self.symbol,
            "day": self.day_key,
            "status": status,
        }
        self._send(message, meta=meta, subject=subject or self._event_subject(status=status))

    def _startup_notify_if_possible(self) -> None:
        if self.ntf_startup or getattr(self, "notifier", None) is None:
            return
        self._send_event(
            status="BOT STARTED",
            emoji="🤖",
            details={
                "Timeframe": self.timeframe,
                "Volume": self.volume,
                "Deviation": self.deviation,
                "Box Window": "22:00-02:00 UTC",
                "Calculation Window": "02:00-03:00 UTC",
                "Profit Flip Window": self._profit_flip_window_label(),
                "TP Model": "Real box 1:1",
                "Channels": ", ".join(self.notify_channels),
            },
            note="Strategy is online. It will calculate the London box, wait for closed-candle breakout, and manage extra flip conditions.",
            subject=f"[{self.bot_name}] {self.symbol} — Bot Started",
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
        self.msg_profit_flip_watch = False
        self.msg_profit_flip_wait_session = False
        self.msg_profit_flip_expired = False
        self.msg_cant_trade_today = False
        self.ntf_wait_box = False
        self.ntf_box_done = False
        self.ntf_wait_break = False
        self.ntf_breakout = False
        self.ntf_wait_rev = False
        self.ntf_missed_rev = False
        self.ntf_profit_flip_watch = False
        self.ntf_profit_flip_wait_session = False
        self.ntf_profit_flip_expired = False
        self.ntf_rev_trigger = False
        self.ntf_cant_trade = False

    def _reset_for_new_day(self, *, k: str) -> None:
        self.box = None
        self.day_key = k
        self.break_side01 = None
        self.break_executed = False
        self.rev_executed = False
        self.rev_missed_today = False
        self.profit_flip_watch = False
        self.profit_flip_expired = False
        self.break_tickets = []
        self.rev_tickets = []
        self.break_tp_ticket = None
        self.break_runner_ticket = None
        self.flip_tp_ticket = None
        self.flip_runner_ticket = None
        self.pending_tp_updates = {}
        self.runner_trail_state = {}
        self.breakout_done_reason = None
        self.break_close_reason = {}
        self.closed_position_ntf_sent = []
        self.day_summary_sent = False
        self.opened_position_ntf_sent = []
        self.tp_inserted_ntf_sent = []
        self.tp_pending_ntf_sent = []
        self.order_failure_ntf_sent = []
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
                self._send_event(
                    status="WAITING FOR BOX WINDOW",
                    emoji="🕒",
                    details={
                        "Box Window": "22:00-02:00 UTC",
                        "Calculation Window": "02:00-03:00 UTC",
                    },
                    note="Waiting for the allowed calculation window before building the London box.",
                    subject=f"[{self.bot_name}] {self.symbol} — Waiting For Box",
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
                self._send_event(
                    status="TRADING SKIPPED",
                    emoji="⛔",
                    details={
                        "Reason": "Missed box calculation window",
                        "Next Window": next_calc.strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "Time Remaining": f"{h:02d}:{m:02d}:{s:02d}",
                    },
                    note="The bot will not trade this symbol today because the London box was not calculated inside the allowed window.",
                    subject=f"[{self.bot_name}] {self.symbol} — Trading Skipped",
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
            self._send_event(
                status="BOX READY",
                emoji="✅",
                details={
                    "Box Window": "22:00-02:00 UTC",
                    "Calculation Window": "02:00-03:00 UTC",
                },
                levels={
                    "High": body_high,
                    "Low": body_low,
                    "Size": size,
                },
                note="London box has been calculated. Waiting for a closed-candle breakout.",
                subject=f"[{self.bot_name}] {self.symbol} — Box Ready",
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

    def _notify_order_failed(
        self,
        *,
        stage: str,
        leg: str,
        side01: int,
        magic: int,
        sl: float,
        tp: Optional[float],
        comment: str,
        result: Any,
        signal_price: Optional[float] = None,
    ) -> None:
        """Send one clean failure notification per stage/leg/comment/day."""
        key = f"{self.day_key}:{stage}:{leg}:{magic}:{comment}"
        if key in self.order_failure_ntf_sent:
            return
        self.order_failure_ntf_sent.append(key)

        self._send_event(
            status="ORDER FAILED",
            emoji="❌",
            details={
                "Stage": stage,
                "Leg": leg,
                "Direction": self._side_name(side01),
                "Magic": magic,
                "Volume": self.volume,
                "Order Status": "Rejected or no ticket returned by MT5",
                "MT5 Result": result,
            },
            levels={
                "Signal Price": signal_price,
                "Stop Loss": sl,
                "Requested TP": tp,
            },
            note="The bot attempted to open this position, but MT5 did not confirm a valid ticket. The strategy may retry if its rules are still valid.",
            subject=f"[{self.bot_name}] {self.symbol} — Order Failed",
        )

    def _notify_position_opened(
        self,
        *,
        p: Position,
        stage: str,
        leg: str,
        side01: int,
        magic: int,
        sl: float,
        tp: Optional[float],
        signal_price: Optional[float] = None,
    ) -> None:
        """Notify only after MT5 confirms the position ticket."""
        try:
            ticket_i = int(getattr(p, "ticket", 0) or 0)
        except Exception:
            ticket_i = 0
        if ticket_i <= 0 or ticket_i in self.opened_position_ntf_sent:
            return

        entry = self._actual_entry_price(p)
        self.opened_position_ntf_sent.append(ticket_i)

        note = "MT5 confirmed the position was opened successfully."
        if str(leg).lower() == "tp":
            note += " TP will now be inserted using the confirmed actual entry price."
        else:
            note += " This runner leg has no fixed TP; it will be managed by the real-box trailing logic."

        self._send_event(
            status="POSITION OPENED",
            emoji="✅",
            details={
                "Stage": stage,
                "Leg": leg,
                "Direction": self._side_name(side01),
                "Magic": magic,
                "Volume": self.volume,
                "Order Status": "Confirmed by MT5",
                "TP Model": "Real box 1:1" if str(leg).lower() == "tp" else "Runner only",
            },
            levels={
                "Signal Price": signal_price,
                "Actual Entry": entry,
                "Stop Loss": sl,
                "Requested TP": tp,
            },
            tickets={
                "Ticket": ticket_i,
            },
            note=note,
            subject=f"[{self.bot_name}] {self.symbol} — {stage} {leg} Opened",
        )

    def _open_position(
        self,
        *,
        side01: int,
        magic: int,
        sl: float,
        tp: Optional[float],
        comment: str,
        stage: str = "TRADE",
        leg: str = "POSITION",
        signal_price: Optional[float] = None,
    ) -> Optional[Position]:
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
            self._notify_order_failed(
                stage=stage,
                leg=leg,
                side01=side01,
                magic=magic,
                sl=sl,
                tp=tp,
                comment=comment,
                result=res,
                signal_price=signal_price,
            )
            return None

        if p.price_open is None:
            try:
                p.refresh_from_mt5_position()
            except Exception as e:
                self.logger.warning(f"[{self.symbol}] post-open refresh failed ticket={p.ticket}: {e}")

        self._notify_position_opened(
            p=p,
            stage=stage,
            leg=leg,
            side01=side01,
            magic=magic,
            sl=sl,
            tp=tp,
            signal_price=signal_price,
        )
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

    def _initial_sl_for_side(self, side01: int) -> Optional[float]:
        """
        Original SL side for the strategy.
        BUY  -> box low
        SELL -> box high

        This is used to calculate the real trade risk from actual entry,
        even after the live SL has been moved by trailing.
        """
        if self.box is None:
            return None
        return float(self.box.low if int(side01) == 0 else self.box.high)

    def _trade_risk_distance(
        self,
        *,
        entry: float,
        side01: int,
        fallback_sl: Optional[float] = None,
    ) -> Optional[float]:
        """
        Real trade box / 1R distance.

        This uses actual MT5 entry price and the original stop-loss side,
        not self.box.size. This protects the strategy from slippage.
        """
        try:
            entry_f = float(entry)
        except Exception:
            return None

        original_sl = self._initial_sl_for_side(int(side01))
        if original_sl is None and fallback_sl is not None:
            try:
                original_sl = float(fallback_sl)
            except Exception:
                original_sl = None

        if original_sl is None:
            return None

        risk = abs(entry_f - float(original_sl))
        return float(risk) if risk > 0 else None

    def _runner_first_trigger_point(self, *, entry: float, side01: int, risk: float) -> float:
        return float(entry) + float(risk) if int(side01) == 0 else float(entry) - float(risk)

    def _runner_next_trigger_point(self, *, previous_trigger_point: float, side01: int, risk: float) -> float:
        return float(previous_trigger_point) + float(risk) if int(side01) == 0 else float(previous_trigger_point) - float(risk)

    def _runner_trigger_reached(self, *, price_current: float, trigger_point: float, side01: int) -> bool:
        return float(price_current) >= float(trigger_point) if int(side01) == 0 else float(price_current) <= float(trigger_point)

    def _runner_sl_is_at_or_beyond_entry(self, *, current_sl: float, entry: float, side01: int) -> bool:
        # BUY: SL at/above entry means break-even or profit locked.
        # SELL: SL at/below entry means break-even or profit locked.
        return float(current_sl) >= float(entry) if int(side01) == 0 else float(current_sl) <= float(entry)

    def _runner_state_for_ticket(
        self,
        *,
        ticket: int,
        entry: float,
        side01: int,
        risk: float,
        current_sl: float,
    ) -> Dict[str, Any]:
        """
        Return or rebuild runner trailing state.

        If the bot restarts and the runner SL is already at/beyond entry,
        we rebuild state as if the first 1R trigger already happened.
        """
        ticket_i = int(ticket)
        state = self.runner_trail_state.get(ticket_i)
        if isinstance(state, dict):
            return state

        first_trigger = self._runner_first_trigger_point(entry=entry, side01=side01, risk=risk)
        be_done = self._runner_sl_is_at_or_beyond_entry(current_sl=current_sl, entry=entry, side01=side01)

        state = {
            "be_done": bool(be_done),
            "previous_trigger_point": float(first_trigger if be_done else 0.0),
            "updates": 1 if be_done else 0,
        }
        self.runner_trail_state[ticket_i] = state
        return state

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

    def _notify_tp_pending(
        self,
        *,
        ticket: int,
        side01: int,
        sl: float,
        context: str,
        reason: str,
        entry: Optional[float] = None,
        result: Any = None,
    ) -> None:
        try:
            ticket_i = int(ticket)
        except Exception:
            ticket_i = 0
        if ticket_i <= 0:
            return

        # Do not spam the same pending alert every loop. If retry later succeeds,
        # _notify_tp_inserted will still send a success alert.
        if ticket_i in self.tp_pending_ntf_sent:
            return
        self.tp_pending_ntf_sent.append(ticket_i)

        self._send_event(
            status="TAKE PROFIT PENDING",
            emoji="⚠️",
            details={
                "Stage": context,
                "Direction": self._side_name(side01),
                "Reason": reason,
                "Action": "Saved for retry",
                "MT5 Result": result,
            },
            levels={
                "Actual Entry": entry,
                "Stop Loss": sl,
            },
            tickets={
                "Ticket": ticket_i,
            },
            note="The position is open, but TP was not inserted yet. The bot saved the ticket and will retry until MT5 confirms the TP or the position closes.",
            subject=f"[{self.bot_name}] {self.symbol} — TP Pending Retry",
        )

    def _notify_tp_inserted(
        self,
        *,
        ticket: int,
        side01: int,
        context: str,
        entry: float,
        sl: float,
        risk_distance: float,
        tp: float,
        retry: bool = False,
    ) -> None:
        try:
            ticket_i = int(ticket)
        except Exception:
            ticket_i = 0
        if ticket_i <= 0 or ticket_i in self.tp_inserted_ntf_sent:
            return
        self.tp_inserted_ntf_sent.append(ticket_i)

        self._send_event(
            status="TAKE PROFIT INSERTED" if not retry else "TAKE PROFIT INSERTED ON RETRY",
            emoji="🎯",
            details={
                "Stage": context,
                "Direction": self._side_name(side01),
                "TP Model": "Real box 1:1",
                "Status": "Confirmed by MT5",
                "Retry": "Yes" if retry else "No",
            },
            levels={
                "Actual Entry": entry,
                "Stop Loss": sl,
                "Real Box / 1R": risk_distance,
                "Take Profit": tp,
            },
            tickets={
                "Ticket": ticket_i,
            },
            note="TP has been inserted using the actual MT5 entry-to-original-SL distance, so the TP is true 1:1 even if slippage changed the entry.",
            subject=f"[{self.bot_name}] {self.symbol} — TP Inserted",
        )

    def _set_tp_from_actual_entry(self, p: Position, *, side01: int, sl: float, context: str, remember_pending: bool = True) -> bool:
        if self.box is None:
            return False
        ticket_i = int(getattr(p, "ticket", 0) or 0)
        entry = self._actual_entry_price(p)
        sl_f = float(sl)

        if entry is None:
            self.logger.warning(f"[{self.symbol}] {context}: cannot set TP; actual entry missing ticket={ticket_i}")
            if remember_pending and ticket_i > 0:
                self._remember_pending_tp(ticket_i, side01=side01, sl=sl_f, context=context, magic=getattr(p, "magic", 0))
                self._notify_tp_pending(
                    ticket=ticket_i,
                    side01=side01,
                    sl=sl_f,
                    context=context,
                    reason="Actual entry price missing after MT5 open confirmation",
                    entry=None,
                    result=None,
                )
            return False

        # Real 1:1 risk model:
        # risk distance = actual MT5 entry to original SL side.
        risk_distance = self._trade_risk_distance(
            entry=float(entry),
            side01=int(side01),
            fallback_sl=sl_f,
        )
        if risk_distance is None or risk_distance <= 0:
            self.logger.warning(
                f"[{self.symbol}] {context}: cannot set TP; invalid real risk distance "
                f"ticket={ticket_i} entry={entry:.5f} sl={sl_f:.5f}"
            )
            if remember_pending and ticket_i > 0:
                self._remember_pending_tp(ticket_i, side01=side01, sl=sl_f, context=context, magic=getattr(p, "magic", 0))
                self._notify_tp_pending(
                    ticket=ticket_i,
                    side01=side01,
                    sl=sl_f,
                    context=context,
                    reason="Invalid real box / risk distance",
                    entry=float(entry),
                    result=None,
                )
            return False

        # 1:1 TP model: TP distance equals actual entry-to-original-SL distance.
        tp = float(entry) + float(risk_distance) if int(side01) == 0 else float(entry) - float(risk_distance)
        p.side01 = int(side01)
        p.sl = sl_f
        p.tp = float(tp)
        res = p.modify_sl_tp(sl=sl_f, tp=float(tp))
        if not res.get("ok"):
            self.logger.warning(f"[{self.symbol}] {context}: TP modify failed ticket={ticket_i} result={res}")
            if remember_pending and ticket_i > 0:
                self._remember_pending_tp(ticket_i, side01=side01, sl=sl_f, context=context, magic=getattr(p, "magic", 0))
                self._notify_tp_pending(
                    ticket=ticket_i,
                    side01=side01,
                    sl=sl_f,
                    context=context,
                    reason="MT5 modify SL/TP failed",
                    entry=float(entry),
                    result=res,
                )
            return False

        self.pending_tp_updates.pop(ticket_i, None)
        self.logger.info(
            f"[{self.symbol}] {context}: TP set using real trade box ticket={ticket_i} "
            f"entry={entry:.5f} sl={sl_f:.5f} real_risk={risk_distance:.5f} tp={tp:.5f}"
        )

        self._notify_tp_inserted(
            ticket=ticket_i,
            side01=side01,
            context=context,
            entry=float(entry),
            sl=sl_f,
            risk_distance=float(risk_distance),
            tp=float(tp),
            retry=str(context).lower().startswith("retry"),
        )
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

    def _float_value(self, value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    def _pnl_text(self, value: Any) -> str:
        x = self._float_value(value)
        if x is None:
            return "N/A"
        sign = "+" if x > 0 else ""
        return f"{sign}{x:.2f}"

    def _result_from_pnl(self, value: Any) -> str:
        x = self._float_value(value)
        if x is None:
            return "UNKNOWN"
        if x > 0:
            return "PROFIT"
        if x < 0:
            return "LOSS"
        return "BREAKEVEN"

    def _emoji_for_result(self, result: str) -> str:
        r = str(result or "").upper()
        if r == "PROFIT":
            return "✅"
        if r == "LOSS":
            return "❌"
        if r == "BREAKEVEN":
            return "➖"
        return "ℹ️"

    def _magic_role(self, magic: int) -> Dict[str, str]:
        try:
            m = int(magic or 0)
        except Exception:
            m = 0
        if m == int(self.magic_break_tp):
            return {"Stage": "Breakout", "Leg": "TP leg"}
        if m == int(self.magic_break_runner):
            return {"Stage": "Breakout", "Leg": "Runner leg"}
        if m == int(self.magic_rev_tp):
            return {"Stage": "Flip", "Leg": "TP leg"}
        if m == int(self.magic_rev_runner):
            return {"Stage": "Flip", "Leg": "Runner leg"}
        return {"Stage": "Unknown", "Leg": "Unknown"}

    def _known_owned_tickets(self) -> List[int]:
        out: List[int] = []
        for t in [
            self.break_tp_ticket,
            self.break_runner_ticket,
            self.flip_tp_ticket,
            self.flip_runner_ticket,
            *self.break_tickets,
            *self.rev_tickets,
        ]:
            try:
                ti = int(t) if t is not None else 0
            except Exception:
                ti = 0
            if ti > 0 and ti not in out:
                out.append(ti)
        return out

    def _same_day_rows(self, *, status: Optional[str] = None, limit: int = 5000) -> List[Dict[str, Any]]:
        rows = self._fetch_position_rows(status=status, limit=limit)
        out: List[Dict[str, Any]] = []
        for row in rows:
            try:
                comment = str(row.get("comment") or "")
            except Exception:
                comment = ""
            if self._comment_matches_day(comment):
                out.append(row)
        return out

    def _finalize_known_closed_positions(self) -> None:
        """
        MT5 may close a position before the DB/cache row is finalized.
        This checks known tickets, and when a ticket is no longer live, it tries
        to pull the final close details from MT5 history so notifications can show
        close price and profit/loss cleanly.
        """
        for ticket in self._known_owned_tickets():
            try:
                if self._refresh_ticket_open(ticket) is not None:
                    continue
            except Exception:
                continue

            row = self._db_row_for_ticket(ticket) or {}
            try:
                side01 = int(row.get("type", self.break_side01 if self.break_side01 is not None else 0) or 0)
            except Exception:
                side01 = int(self.break_side01) if self.break_side01 is not None else 0
            try:
                magic = int(row.get("magic", 0) or 0)
            except Exception:
                magic = 0
            self._history_close_for_ticket(ticket, side01=side01, magic=magic)

    def _close_snapshot_from_row(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            ticket = int(row.get("ticket", 0) or 0)
        except Exception:
            ticket = 0
        if ticket <= 0:
            return None

        try:
            magic = int(row.get("magic", 0) or 0)
        except Exception:
            magic = 0
        try:
            side01 = int(row.get("type", self.break_side01 if self.break_side01 is not None else 0) or 0)
        except Exception:
            side01 = int(self.break_side01) if self.break_side01 is not None else 0

        entry = self._float_value(row.get("price_open"))
        sl = self._float_value(row.get("sl"))
        tp = self._float_value(row.get("tp"))
        close_price = self._float_value(row.get("close_price"))
        total_pnl = self._float_value(row.get("total_pnl"))
        profit = self._float_value(row.get("profit"))
        pnl = total_pnl if total_pnl is not None else profit

        close_time = (
            row.get("close_time")
            or row.get("time_close")
            or row.get("closed_at")
            or row.get("updated_at")
            or "N/A"
        )

        if close_price is None or pnl is None:
            hist = self._history_close_for_ticket(ticket, side01=side01, magic=magic)
            if hist:
                close_price = close_price if close_price is not None else self._float_value(hist.get("close_price"))
                pnl = pnl if pnl is not None else self._float_value(hist.get("total_pnl"))
                if pnl is None:
                    pnl = self._float_value(hist.get("profit"))
                close_time = hist.get("close_time") or close_time

        result = self._result_from_pnl(pnl)
        role = self._magic_role(magic)
        return {
            "ticket": ticket,
            "magic": magic,
            "side01": side01,
            "direction": self._side_name(side01),
            "stage": role.get("Stage", "Unknown"),
            "leg": role.get("Leg", "Unknown"),
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "close_price": close_price,
            "pnl": pnl,
            "result": result,
            "close_time": close_time,
            "comment": str(row.get("comment") or ""),
        }

    def _notify_closed_positions_if_any(self) -> None:
        self._finalize_known_closed_positions()

        for row in self._same_day_rows(status="closed", limit=5000):
            snap = self._close_snapshot_from_row(row)
            if not snap:
                continue
            ticket = int(snap["ticket"])
            if ticket in self.closed_position_ntf_sent:
                continue

            result = str(snap.get("result") or "UNKNOWN").upper()
            emoji = self._emoji_for_result(result)
            self._send_event(
                status=f"POSITION CLOSED — {result}",
                emoji=emoji,
                details={
                    "Stage": snap.get("stage"),
                    "Leg": snap.get("leg"),
                    "Direction": snap.get("direction"),
                    "Result": result,
                    "Profit/Loss": self._pnl_text(snap.get("pnl")),
                    "Magic": snap.get("magic"),
                    "Close Time": snap.get("close_time"),
                    "Comment": snap.get("comment"),
                },
                levels={
                    "Entry": snap.get("entry"),
                    "Stop Loss": snap.get("sl"),
                    "Take Profit": snap.get("tp"),
                    "Close Price": snap.get("close_price"),
                },
                tickets={"Ticket": ticket},
                note="A managed position has closed. The result above shows whether it closed in profit, loss, breakeven, or unknown if broker history was not yet available.",
                subject=f"[{self.bot_name}] {self.symbol} — Position Closed {result}",
            )
            self.closed_position_ntf_sent.append(ticket)

    def _has_owned_open_positions(self) -> bool:
        live = self._live_positions_by_magics(self.owned_magics())
        if live is None:
            return True
        return bool(live)

    def _strategy_day_is_finished(self, now: datetime) -> bool:
        if self._has_owned_open_positions():
            return False
        if self.cant_trade_day_key == self.day_key:
            return True
        if self.rev_executed:
            return True
        if self.rev_missed_today or self.profit_flip_expired:
            return True
        return False

    def _send_day_summary_if_finished(self, now: datetime) -> None:
        if self.day_summary_sent:
            return
        if not self._strategy_day_is_finished(now):
            return

        rows = self._same_day_rows(status="closed", limit=5000)
        snapshots: List[Dict[str, Any]] = []
        for row in rows:
            snap = self._close_snapshot_from_row(row)
            if snap:
                snapshots.append(snap)

        total_positions = len(snapshots)
        pnl_values = [self._float_value(x.get("pnl")) for x in snapshots]
        pnl_numbers = [x for x in pnl_values if x is not None]
        total_pnl = sum(pnl_numbers) if pnl_numbers else None
        wins = sum(1 for x in pnl_numbers if x > 0)
        losses = sum(1 for x in pnl_numbers if x < 0)
        breakeven = sum(1 for x in pnl_numbers if x == 0)
        result = self._result_from_pnl(total_pnl)

        if total_positions == 0:
            return

        self._send_event(
            status=f"DAY CLOSED — {result}",
            emoji=self._emoji_for_result(result),
            details={
                "Trading State": "Finished for this symbol/day",
                "Total Closed Positions": total_positions,
                "Wins": wins,
                "Losses": losses,
                "Breakeven": breakeven,
                "Total Profit/Loss": self._pnl_text(total_pnl),
                "Breakout Status": self._fmt_value(self.breakout_done_reason),
                "Extra Flip Watch": "Expired" if self.profit_flip_expired else ("Used" if self.rev_executed else "Not used"),
                "No More Trades Today": "Yes",
            },
            levels={
                "Box High": self.box.high if self.box else None,
                "Box Low": self.box.low if self.box else None,
                "Box Size": self.box.size if self.box else None,
            },
            note="End-of-day recap: all known managed positions for this symbol are closed, and the strategy will not open another trade for this day.",
            subject=f"[{self.bot_name}] {self.symbol} — Day Closed {result}",
        )
        self.day_summary_sent = True

    def _profit_flip_window_label(self) -> str:
        return f"{self.profit_flip_start_hour:02d}:00-{self.profit_flip_end_hour:02d}:00 UTC"

    def _profit_flip_session_state(self, now: datetime) -> str:
        """
        Returns: before | inside | after.
        Default window is London session in UTC: 07:00 -> 16:00.
        If start == end, the window is treated as all day.
        """
        start = int(self.profit_flip_start_hour) % 24
        end = int(self.profit_flip_end_hour) % 24
        if start == end:
            return "inside"

        h = float(now.hour) + (float(now.minute) / 60.0) + (float(now.second) / 3600.0)

        # Normal same-day session, e.g. 07:00 -> 16:00 UTC.
        if start < end:
            if h < start:
                return "before"
            if h < end:
                return "inside"
            return "after"

        # Overnight session support, e.g. 22:00 -> 02:00 UTC.
        # In the middle gap, it is "before" the next session, not after.
        if h >= start or h < end:
            return "inside"
        return "before"

    def _open_flip_legs(self, *, side01: int, sl: float, signal_price: float, context: str) -> bool:
        if self.box is None:
            return False

        side_name = "SELL" if int(side01) == 1 else "BUY"
        self.logger.info(
            f"[{self.symbol}] {context} {side_name} signal_price={signal_price:.5f} "
            f"sl={sl:.5f} -> opening 2 legs"
        )

        if not self.ntf_rev_trigger:
            self._send_event(
                status="FLIP TRIGGERED",
                emoji="🔁",
                details={
                    "Direction": side_name,
                    "Reason": context,
                    "Legs": "TP leg + Runner leg",
                    "TP Model": "Real box 1:1",
                },
                levels={
                    "Signal Price": signal_price,
                    "Stop Loss": sl,
                    "Box High": self.box.high,
                    "Box Low": self.box.low,
                },
                note="Opposite closed-candle confirmation received. Opening flip legs.",
                subject=f"[{self.bot_name}] {self.symbol} — Flip {side_name}",
            )
            self.ntf_rev_trigger = True

        r1 = self._open_position(
            side01=side01,
            magic=self.magic_rev_tp,
            sl=sl,
            tp=None,
            comment=self._comment(tag="FLIP", leg="TP"),
            stage="FLIP",
            leg="TP",
            signal_price=signal_price,
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
            stage="FLIP",
            leg="RUNNER",
            signal_price=signal_price,
        )
        if r2 is not None and r2.ticket is not None:
            self._remember_leg_ticket(stage="flip", leg="runner", ticket=int(r2.ticket))

        if self.rev_tickets:
            self.rev_executed = True
            return True

        self.logger.warning(f"[{self.symbol}] flip order attempt failed: no legs opened; will retry")
        return False

    def _handle_profit_cycle_possible_flip(self, now: datetime) -> None:
        """
        Called only after _breakout_close_status() returns "tp".
        Because _breakout_close_status() returns "open" while any breakout leg is still live,
        reaching this function means BOTH breakout legs are already closed.
        """
        if self.box is None or self.break_side01 is None:
            return

        if not self.profit_flip_watch:
            self.profit_flip_watch = True
            if not self.msg_profit_flip_watch:
                self.logger.info(
                    f"[{self.symbol}] breakout cycle closed in profit; watching possible flip until "
                    f"{self._profit_flip_window_label()}"
                )
                self.msg_profit_flip_watch = True
            if not self.ntf_profit_flip_watch:
                self._send_event(
                    status="PROFIT FLIP WATCH ACTIVE",
                    emoji="👀",
                    details={
                        "Reason": "Breakout cycle closed in profit",
                        "Watch Window": self._profit_flip_window_label(),
                        "Trigger": "Closed candle beyond opposite box side",
                    },
                    levels={
                        "Box High": self.box.high,
                        "Box Low": self.box.low,
                    },
                    note="Both breakout legs are closed. Bot is watching for one possible extra flip until the London session ends.",
                    subject=f"[{self.bot_name}] {self.symbol} — Profit Flip Watch",
                )
                self.ntf_profit_flip_watch = True

        session_state = self._profit_flip_session_state(now)
        if session_state == "before":
            if not self.msg_profit_flip_wait_session:
                self.logger.info(
                    f"[{self.symbol}] profit-cycle possible flip armed; waiting for session window "
                    f"{self._profit_flip_window_label()}"
                )
                self.msg_profit_flip_wait_session = True
            if not self.ntf_profit_flip_wait_session:
                self._send_event(
                    status="PROFIT FLIP ARMED",
                    emoji="⏳",
                    details={
                        "Watch Window": self._profit_flip_window_label(),
                        "Current State": "Before session",
                        "Trigger": "Closed candle beyond opposite box side",
                    },
                    levels={
                        "Box High": self.box.high,
                        "Box Low": self.box.low,
                    },
                    note="Profit-cycle possible flip is armed, but the session window has not started yet.",
                    subject=f"[{self.bot_name}] {self.symbol} — Profit Flip Armed",
                )
                self.ntf_profit_flip_wait_session = True
            return

        if session_state == "after":
            self.profit_flip_expired = True
            self.rev_missed_today = True
            if not self.msg_profit_flip_expired:
                self.logger.info(
                    f"[{self.symbol}] profit-cycle possible flip expired; session ended "
                    f"({self._profit_flip_window_label()}); no more trades today"
                )
                self.msg_profit_flip_expired = True
            if not self.ntf_profit_flip_expired:
                self._send_event(
                    status="PROFIT FLIP EXPIRED",
                    emoji="⌛",
                    details={
                        "Reason": "London session ended",
                        "Watch Window": self._profit_flip_window_label(),
                        "Action": "No more trades today",
                    },
                    levels={
                        "Box High": self.box.high,
                        "Box Low": self.box.low,
                    },
                    note="No extra flip was triggered before the session expired.",
                    subject=f"[{self.bot_name}] {self.symbol} — Profit Flip Expired",
                )
                self.ntf_profit_flip_expired = True
            return

        # Inside the profit-flip session: wait for a CLOSED candle beyond the opposite box side.
        if not self._flip_triggered():
            return

        side01 = 1 if int(self.break_side01) == 0 else 0
        sl = self.box.high if side01 == 1 else self.box.low

        tick = self._tick()
        if tick is None or not self._spread_ok(tick, context="profit-cycle flip entry"):
            return
        signal_price = tick["bid"] if side01 == 1 else tick["ask"]

        self._open_flip_legs(
            side01=side01,
            sl=sl,
            signal_price=signal_price,
            context="profit-cycle possible flip triggered",
        )

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

        for ticket in tickets:
            p = self._position_shell(ticket=int(ticket))
            snap = p.refresh_from_mt5_position()
            if snap is None:
                self.runner_trail_state.pop(int(ticket), None)
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
                current_sl = float(raw.get("sl"))
                side01 = int(raw.get("type", 0) or 0)
            except Exception:
                continue

            if price_current <= 0 or price_open <= 0 or current_sl <= 0:
                continue

            # Real trade box / 1R:
            # actual MT5 entry price -> original SL side.
            # We do NOT use self.box.size here because slippage can change the real risk.
            # We do NOT use current_sl as the main risk because current_sl changes after trailing.
            risk = self._trade_risk_distance(
                entry=price_open,
                side01=side01,
                fallback_sl=current_sl,
            )
            if risk is None or risk <= 0:
                continue

            state = self._runner_state_for_ticket(
                ticket=int(ticket),
                entry=price_open,
                side01=side01,
                risk=float(risk),
                current_sl=current_sl,
            )

            be_done = bool(state.get("be_done", False))
            previous_trigger = float(state.get("previous_trigger_point", 0.0) or 0.0)

            # First runner move:
            # When price reaches 1R, move SL to actual entry price.
            # This locks the runner at break-even first, instead of jumping straight to 50% trail.
            if not be_done:
                first_trigger = self._runner_first_trigger_point(entry=price_open, side01=side01, risk=float(risk))
                if not self._runner_trigger_reached(price_current=price_current, trigger_point=first_trigger, side01=side01):
                    continue

                new_sl = float(price_open)
                if side01 == 0 and new_sl <= current_sl:
                    # If SL is already at/beyond entry because of restart/manual update, only rebuild state.
                    state["be_done"] = True
                    state["previous_trigger_point"] = float(first_trigger)
                    state["updates"] = max(int(state.get("updates", 0) or 0), 1)
                    continue
                if side01 == 1 and new_sl >= current_sl:
                    state["be_done"] = True
                    state["previous_trigger_point"] = float(first_trigger)
                    state["updates"] = max(int(state.get("updates", 0) or 0), 1)
                    continue

                p.magic = magic
                p.side01 = side01
                p.sl = float(new_sl)
                res = p.modify_sl_tp(sl=float(new_sl), tp=None)
                if res.get("ok"):
                    state["be_done"] = True
                    state["previous_trigger_point"] = float(first_trigger)
                    state["updates"] = 1
                    self.runner_trail_state[int(ticket)] = state

                    self.logger.info(
                        f"[{self.symbol}] runner moved to break-even ticket={ticket} magic={magic} "
                        f"entry={price_open:.5f} current={price_current:.5f} "
                        f"risk={risk:.5f} trigger={first_trigger:.5f} old_sl={current_sl:.5f} new_sl={new_sl:.5f}"
                    )
                    self._send_event(
                        status="RUNNER MOVED TO BREAK-EVEN",
                        emoji="🟰",
                        details={
                            "Stage": self._magic_role(magic).get("Stage"),
                            "Leg": self._magic_role(magic).get("Leg"),
                            "Direction": self._side_name(side01),
                            "Magic": magic,
                            "Trail Model": "First 1R -> break-even",
                        },
                        levels={
                            "Entry": price_open,
                            "Current Price": price_current,
                            "Real Trade Box / 1R": risk,
                            "Trigger Point": first_trigger,
                            "Old SL": current_sl,
                            "New SL": new_sl,
                        },
                        tickets={"Ticket": ticket},
                        note="Runner reached 1R. SL moved to actual entry price first, so the runner can no longer close in loss.",
                        subject=f"[{self.bot_name}] {self.symbol} — Runner Break-Even",
                    )
                continue

            # Later runner moves:
            # Use previous_trigger_point, not current SL, so the first break-even move does not
            # immediately cause a 50% trail update in the same price area.
            if previous_trigger <= 0:
                previous_trigger = self._runner_first_trigger_point(entry=price_open, side01=side01, risk=float(risk))
                state["previous_trigger_point"] = float(previous_trigger)

            next_trigger = self._runner_next_trigger_point(
                previous_trigger_point=previous_trigger,
                side01=side01,
                risk=float(risk),
            )

            if not self._runner_trigger_reached(price_current=price_current, trigger_point=next_trigger, side01=side01):
                continue

            # Move SL by 50% of the real trade box from the last trigger point.
            # BUY example: last trigger 1R, next trigger 2R -> new SL = 1.5R.
            # SELL example: last trigger -1R, next trigger -2R -> new SL = -1.5R.
            new_sl = previous_trigger + (float(risk) * 0.5) if side01 == 0 else previous_trigger - (float(risk) * 0.5)

            # Never make SL worse.
            if side01 == 0 and new_sl <= current_sl:
                state["previous_trigger_point"] = float(next_trigger)
                self.runner_trail_state[int(ticket)] = state
                continue
            if side01 == 1 and new_sl >= current_sl:
                state["previous_trigger_point"] = float(next_trigger)
                self.runner_trail_state[int(ticket)] = state
                continue

            p.magic = magic
            p.side01 = side01
            p.sl = float(new_sl)
            res = p.modify_sl_tp(sl=float(new_sl), tp=None)
            if res.get("ok"):
                state["previous_trigger_point"] = float(next_trigger)
                state["updates"] = int(state.get("updates", 1) or 1) + 1
                self.runner_trail_state[int(ticket)] = state

                self.logger.info(
                    f"[{self.symbol}] runner stepped trail ticket={ticket} magic={magic} "
                    f"entry={price_open:.5f} current={price_current:.5f} risk={risk:.5f} "
                    f"prev_trigger={previous_trigger:.5f} next_trigger={next_trigger:.5f} "
                    f"old_sl={current_sl:.5f} new_sl={new_sl:.5f}"
                )
                self._send_event(
                    status="RUNNER TRAILED",
                    emoji="🛡️",
                    details={
                        "Stage": self._magic_role(magic).get("Stage"),
                        "Leg": self._magic_role(magic).get("Leg"),
                        "Direction": self._side_name(side01),
                        "Magic": magic,
                        "Trail Model": "Previous trigger + 1R -> move SL by 0.5R",
                        "Trail Updates": state.get("updates"),
                    },
                    levels={
                        "Entry": price_open,
                        "Current Price": price_current,
                        "Real Trade Box / 1R": risk,
                        "Previous Trigger": previous_trigger,
                        "Next Trigger": next_trigger,
                        "Old SL": current_sl,
                        "New SL": new_sl,
                    },
                    tickets={"Ticket": ticket},
                    note="Runner price moved another 1R from the saved previous trigger point. SL stepped forward by 0.5R.",
                    subject=f"[{self.bot_name}] {self.symbol} — Runner Trailed",
                )

    def step(self):
        now = self._now()
        self._startup_notify_if_possible()
        self._calculate_box_if_needed(now)

        if self.box is None:
            return
        if self.cant_trade_day_key == self.day_key:
            self._notify_closed_positions_if_any()
            self._send_day_summary_if_finished(now)
            return

        if self._state_recovered_day_key != self.day_key:
            open_recovered = self._recover_open_trade_state()
            closed_recovered = self._recover_closed_trade_state_from_db()
            if open_recovered and closed_recovered:
                self._state_recovered_day_key = self.day_key

        self._retry_pending_tp_updates()
        self._trail_runners(now)
        self._notify_closed_positions_if_any()
        self._send_day_summary_if_finished(now)

        if self.day_summary_sent:
            return

        # 1) Breakout stage: only once per day.
        if not self.break_executed:
            if not self.msg_wait_break:
                self.logger.info(f"[{self.symbol}] box ready -> waiting for closed-candle breakout")
                self.msg_wait_break = True
            if not self.ntf_wait_break:
                self._send_event(
                    status="WAITING FOR BREAKOUT",
                    emoji="📡",
                    details={
                        "Entry Rule": "Closed candle confirmation",
                        "Buy Trigger": "Close above box high",
                        "Sell Trigger": "Close below box low",
                        "Legs": "TP leg + Runner leg",
                    },
                    levels={
                        "Box High": self.box.high,
                        "Box Low": self.box.low,
                        "Box Size": self.box.size,
                    },
                    note="Box is ready. Bot is waiting for the last closed candle to break outside the box.",
                    subject=f"[{self.bot_name}] {self.symbol} — Waiting For Breakout",
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
                self._send_event(
                    status="BREAKOUT TRIGGERED",
                    emoji="🚀",
                    details={
                        "Direction": self._side_name(side01),
                        "Entry Rule": "Closed candle confirmation",
                        "Legs": "TP leg + Runner leg",
                        "TP Model": "Real box 1:1",
                    },
                    levels={
                        "Signal Price": signal_price,
                        "Box High": self.box.high,
                        "Box Low": self.box.low,
                        "Stop Loss": sl,
                    },
                    note=f"Closed candle confirmed {'above the box high' if side01 == 0 else 'below the box low'}. Opening two breakout legs.",
                    subject=f"[{self.bot_name}] {self.symbol} — Breakout {self._side_name(side01)}",
                )
                self.ntf_breakout = True

            p1 = self._open_position(
                side01=side01,
                magic=self.magic_break_tp,
                sl=sl,
                tp=None,
                comment=self._comment(tag="BREAK", leg="TP"),
                stage="BREAKOUT",
                leg="TP",
                signal_price=signal_price,
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
                stage="BREAKOUT",
                leg="RUNNER",
                signal_price=signal_price,
            )
            if p2 is not None and p2.ticket is not None:
                self._remember_leg_ticket(stage="break", leg="runner", ticket=int(p2.ticket))

            if self.break_tickets:
                self.break_executed = True
            else:
                self.logger.warning(f"[{self.symbol}] breakout order attempt failed: no legs opened; will retry")
            return

        # 2) Flip stage.
        #    A) If breakout closed by SL: normal failed-breakout flip.
        #    B) If breakout cycle closed by TP/profit: wait until both legs are closed,
        #       then allow one possible opposite flip only inside the London session window.
        if self.break_executed and not self.rev_executed and not self.rev_missed_today:
            if not self.msg_wait_rev:
                self.logger.info(f"[{self.symbol}] waiting for flip conditions")
                self.msg_wait_rev = True
            if not self.ntf_wait_rev:
                self._send_event(
                    status="WAITING FOR FLIP CONDITIONS",
                    emoji="🔎",
                    details={
                        "Failed Breakout Flip": "Allowed only after SL close",
                        "Profit Extra Flip": "Allowed after both breakout legs close in profit",
                        "Profit Flip Window": self._profit_flip_window_label(),
                    },
                    levels={
                        "Box High": self.box.high,
                        "Box Low": self.box.low,
                    },
                    note="Bot is checking whether the breakout cycle qualifies for a failed-breakout flip or an extra profit-cycle flip.",
                    subject=f"[{self.bot_name}] {self.symbol} — Waiting For Flip",
                )
                self.ntf_wait_rev = True

            close_status = self._breakout_close_status()
            if close_status == "open":
                return

            if close_status == "tp":
                self._handle_profit_cycle_possible_flip(now)
                return

            if close_status != "sl":
                self.logger.warning(f"[{self.symbol}] breakout close status={close_status}; waiting/retrying before flip")
                return

            # Normal failed-breakout flip: wait for a CLOSED candle beyond the SL/opposite side.
            if not self._flip_triggered():
                return

            side01 = 1 if int(self.break_side01) == 0 else 0
            sl = self.box.high if side01 == 1 else self.box.low

            tick = self._tick()
            if tick is None or not self._spread_ok(tick, context="failed-breakout flip entry"):
                return
            signal_price = tick["bid"] if side01 == 1 else tick["ask"]

            self._open_flip_legs(
                side01=side01,
                sl=sl,
                signal_price=signal_price,
                context="failed-breakout flip triggered",
            )
            return

        return
