#!/usr/bin/env python3
"""
PNT 로그인 REST API 호출 → accessToken 반환.
  python pnt_login.py
  python pnt_login.py --print_token   # 토큰만 출력 (.env 복사용)
"""
import os
import argparse
from dotenv import load_dotenv
import requests

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
if os.getenv("LOAD_PARENT_ENV", "false").strip().lower() in {"1", "true", "yes"}:
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# 로그인은 apichaidev 에서만 허용 (chaidev 는 405)
LOGIN_URL = os.getenv("PNT_LOGIN_URL", "https://apichaidev.atcrk.co.kr/api/v1/auth/login")

 
def get_token(user_id: str = None, password: str = None) -> str:
    """로그인 API 호출 후 accessToken 반환. 실패 시 빈 문자열."""
    user_id = user_id or os.getenv("PNT_USER_ID", "")
    password = password or os.getenv("PNT_USER_PASSWORD", "")
    if not user_id or not password:
        print("[error] PNT_USER_ID, PNT_USER_PASSWORD 를 .env 에 넣거나 인자로 주세요.")
        return ""

    payload = {
        "loginType": "chai",
        "userId": user_id,
        "userPassword": password,
        "ipaddress": "",
    }
    try:
        r = requests.post(LOGIN_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        r.raise_for_status()
        data = r.json()
        token = data.get("accessToken") or data.get("access_token") or ""
        return token
    except requests.RequestException as e:
        print(f"[error] 로그인 실패: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"  응답: {e.response.text[:300]}")
        return ""
    except Exception as e:
        print(f"[error] {e}")
        return ""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--print_token", action="store_true", help="토큰만 출력 (PNT_ACCESS_TOKEN=... 복사용)")
    args = ap.parse_args()

    token = get_token()
    if not token:
        exit(1)
    if args.print_token:
        print(token)
    else:
        print("여기를 .env 에 넣으세요: PNT_ACCESS_TOKEN=...")
        print(f"PNT_ACCESS_TOKEN={token}")
