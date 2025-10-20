# -*- coding: utf-8 -*-

import os
import json
import uuid
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone

import gspread
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("NooLogicCRM")

CONTACTS_SHEET   = "Contacts"
DEALS_SHEET      = "Deals"
ACTIVITIES_SHEET = "Activities"
CLIENTS_SHEET    = "Clients"
MESSAGES_SHEET   = "Messages"

CONTACT_HEADERS = [
    "contact_id", "created_at", "name", "phone_norm", "email",
    "messenger", "handle", "company", "source",
    "utm_campaign", "utm_adset", "utm_content", "notes",
]

DEAL_HEADERS = [
    "deal_id", "created_at", "contact_id", "title", "pipeline",
    "stage", "amount", "currency", "deadline", "owner", "status", "notes",
]

ACTIVITY_HEADERS = [
    "activity_id", "ts", "deal_id", "type", "payload_json", "author",
]

CLIENT_HEADERS = [
    "client_id", "ig_user_id", "name", "phone", "messenger",
    "handle", "email", "source", "first_seen_at", "last_seen_at",
    "contacts_count", "notes",
]

MESSAGE_HEADERS = [
    "msg_id", "client_id", "ig_user_id", "direction", "text",
    "channel", "created_at", "meta_json",
]

DEFAULT_PIPELINE = "lead"
DEFAULT_STAGE    = "lead"
DEFAULT_CURRENCY = "EUR"
DEFAULT_STATUS   = "open"

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def gen_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{prefix.upper()}-{stamp}-{uuid.uuid4().hex[:6]}"

def _str(v: Optional[str]) -> str:
    return (v or "").strip()

def _norm_phone(s: Optional[str]) -> str:
    if not s:
        return ""
    return "".join(ch for ch in str(s) if ch.isdigit())

def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).replace(",", ".").strip()
        return float(s)
    except Exception:
        return default

def _open_sheet_from_env() -> gspread.Spreadsheet:
    sheet_id  = os.getenv("SHEET_ID", "").strip()
    creds_path = os.getenv("GOOGLE_CREDS_PATH", "").strip()
    if not sheet_id:
        raise RuntimeError("SHEET_ID не задано в env")
    if not creds_path or not os.path.exists(creds_path):
        raise RuntimeError(f"GOOGLE_CREDS_PATH не знайдено: {creds_path}")

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = Credentials.from_service_account_file(creds_path, scopes=scopes)
    gc = gspread.authorize(credentials)
    return gc.open_by_key(sheet_id)

def _ensure_worksheet(spread: gspread.Spreadsheet, title: str, headers: List[str]) -> gspread.Worksheet:
    try:
        ws = spread.worksheet(title)
        first_row = ws.row_values(1)
        if first_row != headers:
            ws.resize(rows=1, cols=len(headers))
            ws.update(values=[headers], range_name="1:1")
        return ws
    except gspread.WorksheetNotFound:
        ws = spread.add_worksheet(title=title, rows=1000, cols=len(headers))
        ws.update(values=[headers], range_name="1:1")
        return ws

def _append(ws: gspread.Worksheet, headers: List[str], data: Dict[str, Any]) -> None:
    ws.append_row([str(data.get(h, "")) for h in headers], value_input_option="USER_ENTERED")

def _read_row(ws: gspread.Worksheet, row_idx: int, headers: List[str]) -> Dict[str, str]:
    vals = ws.row_values(row_idx)
    return {h: (vals[i] if i < len(vals) else "") for i, h in enumerate(headers)}

class NooLogicCRM:
    def __init__(self):
        self.spread = _open_sheet_from_env()

        self.ws_contacts   = _ensure_worksheet(self.spread, CONTACTS_SHEET,   CONTACT_HEADERS)
        self.ws_deals      = _ensure_worksheet(self.spread, DEALS_SHEET,      DEAL_HEADERS)
        self.ws_activities = _ensure_worksheet(self.spread, ACTIVITIES_SHEET, ACTIVITY_HEADERS)
        self.ws_clients    = _ensure_worksheet(self.spread, CLIENTS_SHEET,    CLIENT_HEADERS)
        self.ws_messages   = _ensure_worksheet(self.spread, MESSAGES_SHEET,   MESSAGE_HEADERS)

        self.deals_col_idx      = {n: i + 1 for i, n in enumerate(DEAL_HEADERS)}
        self.activities_col_idx = {n: i + 1 for i, n in enumerate(ACTIVITY_HEADERS)}

        log.info("✅ Під’єднано до Google Sheet: %s", self.spread.title)

    def add_contact(self,
                    name: Optional[str],
                    phone: Optional[str],
                    email: Optional[str],
                    messenger: Optional[str],
                    handle: Optional[str],
                    company: Optional[str],
                    source: Optional[str],
                    utm_campaign: Optional[str] = "",
                    utm_adset: Optional[str] = "",
                    utm_content: Optional[str] = "",
                    notes: Optional[str] = "") -> Tuple[str, int]:

        contact_id = gen_id("C")
        values = {
            "contact_id":  contact_id,
            "created_at":  now_iso(),
            "name":        _str(name),
            "phone_norm":  _norm_phone(phone),
            "email":       _str(email),
            "messenger":   _str(messenger),
            "handle":      _str(handle),
            "company":     _str(company),
            "source":      _str(source) or "instagram",
            "utm_campaign": _str(utm_campaign),
            "utm_adset":    _str(utm_adset),
            "utm_content":  _str(utm_content),
            "notes":        _str(notes),
        }
        _append(self.ws_contacts, CONTACT_HEADERS, values)
        row_idx = len(self.ws_contacts.get_all_values())
        return contact_id, row_idx

    def add_client(self,
                   ig_user_id: Optional[str],
                   name: Optional[str],
                   phone: Optional[str],
                   messenger: Optional[str],
                   handle: Optional[str],
                   email: Optional[str],
                   source: Optional[str],
                   notes: Optional[str]) -> str:

        client_id = gen_id("CLI")
        now = now_iso()
        values = {
            "client_id":     client_id,
            "ig_user_id":    _str(ig_user_id),
            "name":          _str(name),
            "phone":         _norm_phone(phone),
            "messenger":     _str(messenger),
            "handle":        _str(handle),
            "email":         _str(email),
            "source":        _str(source) or "instagram",
            "first_seen_at": now,
            "last_seen_at":  now,
            "contacts_count": "1",
            "notes":         _str(notes),
        }
        _append(self.ws_clients, CLIENT_HEADERS, values)
        return client_id

    def create_deal(self,
                    contact_id: str,
                    title: str,
                    pipeline: Optional[str] = DEFAULT_PIPELINE,
                    stage: Optional[str] = DEFAULT_STAGE,
                    amount: Optional[float] = 0.0,
                    currency: Optional[str] = DEFAULT_CURRENCY,
                    deadline: Optional[str] = "",
                    owner: Optional[str] = "",
                    status: Optional[str] = DEFAULT_STATUS,
                    notes: Optional[str] = "") -> str:
        deal_id = gen_id("DEAL")
        values = {
            "deal_id":    deal_id,
            "created_at": now_iso(),
            "contact_id": _str(contact_id),
            "title":      _str(title) or "Без назви",
            "pipeline":   _str(pipeline) or DEFAULT_PIPELINE,
            "stage":      _str(stage) or DEFAULT_STAGE,
            "amount":     str(_to_float(amount, 0.0)),
            "currency":   _str(currency) or DEFAULT_CURRENCY,
            "deadline":   _str(deadline),
            "owner":      _str(owner),
            "status":     _str(status) or DEFAULT_STATUS,
            "notes":      _str(notes),
        }
        _append(self.ws_deals, DEAL_HEADERS, values)
        return deal_id

    def update_deal_stage(self, deal_id: str, new_stage: str, new_status: Optional[str] = None) -> bool:
        try:
            cell = self.ws_deals.find(deal_id, in_column=self.deals_col_idx["deal_id"])
            if not cell or cell.row <= 1:
                return False
            row = _read_row(self.ws_deals, cell.row, DEAL_HEADERS)
            row["stage"] = _str(new_stage)
            if new_status is not None:
                row["status"] = _str(new_status)
            self.ws_deals.update(values=[[row[h] for h in DEAL_HEADERS]],
                                 range_name=f"{cell.row}:{cell.row}")
            return True
        except Exception as e:
            log.exception("update_deal_stage error: %s", e)
            return False

    def log_activity(self, deal_id: str, type_: str, payload: Dict[str, Any], author: str = "bot") -> str:
        activity_id = gen_id("ACT")
        values = {
            "activity_id": activity_id,
            "ts":          now_iso(),
            "deal_id":     _str(deal_id),
            "type":        _str(type_),
            "payload_json": json.dumps(payload, ensure_ascii=False),
            "author":      _str(author),
        }
        _append(self.ws_activities, ACTIVITY_HEADERS, values)
        return activity_id

    def log_message(self,
                    ig_user_id: Optional[str],
                    direction: str,
                    text: str,
                    client_id: Optional[str] = None,
                    channel: str = "instagram",
                    meta: Optional[Dict[str, Any]] = None) -> str:
        msg_id = gen_id("MSG")
        values = {
            "msg_id":     msg_id,
            "client_id":  _str(client_id),
            "ig_user_id": _str(ig_user_id),
            "direction":  "in" if direction not in ("in", "out") else direction,
            "text":       (text or "")[:1000],
            "channel":    _str(channel),
            "created_at": now_iso(),
            "meta_json":  json.dumps(meta or {}, ensure_ascii=False),
        }
        _append(self.ws_messages, MESSAGE_HEADERS, values)
        return msg_id

    def record_contact(self,
                       ig_user_id: Optional[str],
                       info: Dict[str, Optional[str]],
                       channel: str = "instagram",
                       extra: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:

        contact_id, _ = self.add_contact(
            name=info.get("name"),
            phone=info.get("phone"),
            email=info.get("email"),
            messenger=info.get("messenger"),
            handle=info.get("handle"),
            company=(extra or {}).get("company"),
            source=channel,
            utm_campaign=(extra or {}).get("utm_campaign", ""),
            utm_adset=(extra or {}).get("utm_adset", ""),
            utm_content=(extra or {}).get("utm_content", ""),
            notes=(extra or {}).get("notes", ""),
        )

        client_id = self.add_client(
            ig_user_id=ig_user_id,
            name=info.get("name"),
            phone=info.get("phone"),
            messenger=info.get("messenger"),
            handle=info.get("handle"),
            email=info.get("email"),
            source=channel,
            notes=(extra or {}).get("notes", ""),
        )
        return contact_id, client_id

    def find_client_id(self, *_, **__) -> Optional[str]:
        return None

    def find_contact_id(self, *_, **__) -> Optional[str]:
        return None


_CRM_SINGLETON: Optional[NooLogicCRM] = None

def get_crm() -> NooLogicCRM:
    global _CRM_SINGLETON
    if _CRM_SINGLETON is None:
        _CRM_SINGLETON = NooLogicCRM()
    return _CRM_SINGLETON


if __name__ == "__main__":
    crm = get_crm()
    info = {
        "name": "Test User",
        "phone": "+380501112233",
        "email": "test@example.com",
        "messenger": "Telegram",
        "handle": "test_handle",
    }
    contact_id, client_id = crm.record_contact("9999999999", info, channel="instagram",
                                               extra={"company": "NooLogic", "notes": "onboarding demo insert"})
    log.info("Contact=%s Client=%s", contact_id, client_id)
    deal_id = crm.create_deal(contact_id, title="Test Deal", pipeline="lead", stage="lead",
                              amount=3000, currency="EUR", owner="sales", notes="Автотест")
    log.info("Deal=%s", deal_id)
    crm.log_activity(deal_id, "system.note", {"msg": "Created from self-test"}, author="bot")
    crm.log_message("9999999999", "in", "Привіт, це тестове повідомлення", client_id=client_id)
