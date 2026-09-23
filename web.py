"""Local browser dashboard for the VEINN research prototype."""

from __future__ import annotations

import json
import mimetypes
import os
import traceback
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from src.core import (
    decrypt_with_priv,
    decrypt_with_public_veinn,
    encrypt_with_pub,
    encrypt_with_public_veinn,
    veinn_from_seed,
)
from src.models import VeinnParams
from src.utils.keygen import generate_keypair
from src.utils.keystore import create_keystore, store_key_in_keystore

ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web" / "dist"


def _params(payload: dict) -> VeinnParams:
    return VeinnParams(
        n=int(payload["n"]),
        rounds=int(payload["rounds"]),
        layers_per_round=int(payload["layers_per_round"]),
        shuffle_stride=int(payload["shuffle_stride"]),
        use_lwe=bool(payload["use_lwe"]),
        q=int(payload["q"]),
    )


def _optional_nonce(payload: dict) -> bytes | None:
    from base64 import b64decode

    value = str(payload.get("nonce", "")).strip()
    return b64decode(value) if value else None


def run_operation(payload: dict) -> dict:
    operation = payload.get("operation")
    file_type = payload.get("file_type", "json")
    if operation == "encrypt_veinn":
        output = encrypt_with_public_veinn(
            seed_input=str(payload["seed"]),
            file_type=file_type,
            message=payload.get("message") or None,
            in_path=payload.get("in_path") or None,
            vp=_params(payload),
            out_file=str(payload.get("out_file") or "enc_pub_veinn"),
            nonce=_optional_nonce(payload),
            mode=str(payload.get("mode", "cbc")),
        )
        return {"message": "VEINN encryption complete.", "output": output}
    if operation == "encrypt_public":
        output = encrypt_with_pub(
            pubfile=str(payload["pubfile"]),
            file_type=file_type,
            message=payload.get("message") or None,
            in_path=payload.get("in_path") or None,
            vp=_params(payload),
            seed_len=int(payload["seed_len"]),
            nonce=_optional_nonce(payload),
            out_file=str(payload.get("out_file") or "enc_pub"),
            mode=str(payload.get("mode", "cbc")),
        )
        return {"message": "Public-key encryption complete.", "output": output}
    if operation == "decrypt_veinn":
        result = decrypt_with_public_veinn(
            seed_input=str(payload["seed"]),
            file_type=file_type,
            enc_file=str(payload["enc_file"]),
            validity_window=int(payload["validity_window"]),
        )
        return {"message": "VEINN decryption complete.", "output": result["file"], "plaintext": result["text"]}
    if operation == "decrypt_private":
        result = decrypt_with_priv(
            keystore=payload.get("keystore") or None,
            privfile=payload.get("privfile") or None,
            encfile=str(payload.get("encfile") or payload["enc_file"]),
            passphrase=payload.get("passphrase") or None,
            key_name=payload.get("key_name") or None,
            file_type=file_type,
            validity_window=int(payload["validity_window"]),
        )
        return {"message": "Private-key decryption complete.", "output": result["file"], "plaintext": result["text"]}
    if operation == "create_keystore":
        create_keystore(str(payload["passphrase"]), str(payload.get("keystore") or "keystore.json"))
        return {"message": "Encrypted keystore created.", "output": str(payload.get("keystore") or "keystore.json")}
    if operation == "generate_keypair":
        keypair = generate_keypair()
        pubfile = str(payload.get("pubfile") or "public_key.json")
        privfile = str(payload.get("privfile") or "private_key.json")
        with open(pubfile, "w") as file:
            json.dump({"ek": keypair["ek"]}, file)
        if payload.get("store_private") and payload.get("passphrase") and payload.get("key_name"):
            keystore = str(payload.get("keystore") or "keystore.json")
            store_key_in_keystore(str(payload["passphrase"]), str(payload["key_name"]), keypair, keystore)
            return {"message": "Public key generated; private key stored in encrypted keystore.", "output": f"{pubfile}, {keystore}"}
        with open(privfile, "w") as file:
            json.dump(keypair, file)
        return {"message": "Public/private keypair generated.", "output": f"{pubfile}, {privfile}"}
    if operation == "derive_veinn":
        veinn_from_seed(str(payload["seed"]), _params(payload))
        return {"message": "Public VEINN derived from seed.", "output": "VEINN key derived successfully"}
    raise ValueError("Unknown operation.")


def run_self_test() -> dict:
    checks = []
    derived = veinn_from_seed("startup-self-test", VeinnParams(n=64, rounds=1, layers_per_round=1, shuffle_stride=1))
    checks.append({"name": "VEINN key derivation", "passed": bool(derived)})
    with tempfile.TemporaryDirectory(prefix="veinn-self-test-") as directory:
        keystore = os.path.join(directory, "keystore.json")
        create_keystore("startup-test-passphrase", keystore)
        checks.append({"name": "Encrypted keystore creation", "passed": os.path.isfile(keystore)})
        keypair = generate_keypair()
        checks.append({"name": "Kyber keypair generation", "passed": bool(keypair.get("ek") and keypair.get("dk"))})
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


class DashboardHandler(BaseHTTPRequestHandler):
    def _send_json(self, body: dict, status: int = 200) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:8765")
        self.end_headers()
        self.wfile.write(encoded)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:8765")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.end_headers()

    def do_POST(self) -> None:
        if self.path != "/api/operate":
            self._send_json({"error": "Not found."}, 404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size))
            self._send_json(run_operation(payload))
        except Exception as error:
            self._send_json(
                {"error": str(error), "trace": traceback.format_exc(limit=2)},
                400,
            )

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/api/self-test":
            try:
                self._send_json(run_self_test())
            except Exception as error:
                self._send_json({"passed": False, "error": str(error)}, 500)
            return
        relative = "index.html" if route == "/" else route.removeprefix("/")
        requested = (WEB_ROOT / relative).resolve()
        if WEB_ROOT not in requested.parents or not requested.is_file():
            self.send_error(404)
            return
        content = requested.read_bytes()
        self.send_response(200)
        content_type = mimetypes.guess_type(str(requested))[0]
        if requested.suffix == ".js":
            content_type = "text/javascript"
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    os.chdir(ROOT)
    server = ThreadingHTTPServer(("127.0.0.1", 8765), DashboardHandler)
    print("VEINN dashboard: http://127.0.0.1:8765")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
