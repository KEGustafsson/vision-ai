"""Lightweight ONVIF PTZ control.

PTZ domes (the Arabella cameras are Hikvision 4x zoom domes) speak ONVIF, a
SOAP-over-HTTP protocol. The full ``onvif-zeep`` client drags in ``zeep`` and a
large WSDL stack that we don't want on the minimal Jetson image, and we only
need a handful of operations: discover the PTZ/media endpoints, fetch a media
profile token, and issue ContinuousMove / Stop / GotoHomePosition. So this
module talks ONVIF directly with the stdlib (``urllib`` + WS-UsernameToken
digest auth + ElementTree).

Host and credentials default to the ones embedded in the camera's RTSP url
(``rtsp://user:pass@host:554/...``) so the skipper configures them in one place;
``onvif_*`` config fields override per camera when the ONVIF service lives on a
different host/port/account than the RTSP stream.
"""

from __future__ import annotations

import base64
import hashlib
import os
import threading
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .config import CameraConfig, Settings

# ONVIF SOAP namespaces. We match element *local* names when parsing replies so
# we don't have to track each vendor's prefix choices, but requests must carry
# the correct default namespaces or the camera rejects them.
_NS_PTZ = "http://www.onvif.org/ver20/ptz/wsdl"
_NS_MEDIA = "http://www.onvif.org/ver10/media/wsdl"
_NS_DEVICE = "http://www.onvif.org/ver10/device/wsdl"
_NS_SCHEMA = "http://www.onvif.org/ver10/schema"

_HTTP_TIMEOUT_S = 5.0


def _resolve_endpoint(cfg: CameraConfig) -> Optional["_Endpoint"]:
    """Effective ONVIF host/port/credentials for a camera, or None if unusable.

    Falls back to the host + userinfo of the RTSP url so a single configured
    url drives both streaming and PTZ.
    """
    host = cfg.onvif_host
    user = cfg.onvif_user
    password = cfg.onvif_password
    if cfg.url:
        parsed = urllib.parse.urlparse(cfg.url)
        host = host or parsed.hostname
        # unquote so a url-encoded password in the RTSP url still works.
        if user is None and parsed.username:
            user = urllib.parse.unquote(parsed.username)
        if password is None and parsed.password is not None:
            password = urllib.parse.unquote(parsed.password)
    if not host:
        return None
    return _Endpoint(host=host, port=cfg.onvif_port, user=user or "", password=password or "")


class _Endpoint:
    def __init__(self, host: str, port: int, user: str, password: str):
        self.host = host
        self.port = port
        self.user = user
        self.password = password

    @property
    def base(self) -> str:
        return f"http://{self.host}:{self.port}"


def _security_header(user: str, password: str) -> str:
    """WS-Security UsernameToken with a PasswordDigest (ONVIF default auth).

    digest = base64( SHA1( nonce + created + password ) ).
    """
    nonce = os.urandom(16)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    digest = base64.b64encode(
        hashlib.sha1(nonce + created.encode() + password.encode()).digest()
    ).decode()
    nonce_b64 = base64.b64encode(nonce).decode()
    wsse = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"
    wsu = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd"
    pwtype = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest"
    enctype = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary"
    return (
        f'<s:Header><Security s:mustUnderstand="1" xmlns="{wsse}">'
        f"<UsernameToken><Username>{user}</Username>"
        f'<Password Type="{pwtype}">{digest}</Password>'
        f'<Nonce EncodingType="{enctype}">{nonce_b64}</Nonce>'
        f'<Created xmlns="{wsu}">{created}</Created>'
        "</UsernameToken></Security></s:Header>"
    )


def _envelope(body: str, user: str, password: str) -> bytes:
    header = _security_header(user, password) if user else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">'
        f"{header}<s:Body>{body}</s:Body></s:Envelope>"
    ).encode()


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _findall_local(root: ET.Element, name: str) -> List[ET.Element]:
    return [el for el in root.iter() if _localname(el.tag) == name]


class OnvifPtz:
    """ONVIF PTZ client for one camera. Discovery results are cached.

    A lock serialises discovery so concurrent move requests don't each kick off
    a GetServices round-trip.
    """

    def __init__(self, endpoint: _Endpoint):
        self._ep = endpoint
        self._lock = threading.Lock()
        self._ptz_xaddr: Optional[str] = None
        self._profile_token: Optional[str] = None

    def _post(self, xaddr: str, body: str) -> ET.Element:
        data = _envelope(body, self._ep.user, self._ep.password)
        req = urllib.request.Request(
            xaddr,
            data=data,
            headers={"Content-Type": "application/soap+xml; charset=utf-8"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
            return ET.fromstring(resp.read())

    def _rewrite_host(self, xaddr: str) -> str:
        """Point a discovered XAddr back at the configured host:port.

        Cameras sometimes advertise an XAddr with their own internal IP (or the
        wrong port behind NAT). We only trust the path; the host:port is the one
        we already reach successfully.
        """
        path = urllib.parse.urlparse(xaddr).path or "/onvif/ptz_service"
        return f"{self._ep.base}{path}"

    def _discover(self) -> None:
        if self._ptz_xaddr and self._profile_token:
            return
        with self._lock:
            if self._ptz_xaddr and self._profile_token:
                return
            device_xaddr = f"{self._ep.base}/onvif/device_service"
            services = self._post(
                device_xaddr,
                f'<GetServices xmlns="{_NS_DEVICE}">'
                "<IncludeCapability>false</IncludeCapability></GetServices>",
            )
            ptz_xaddr = media_xaddr = None
            for svc in _findall_local(services, "Service"):
                ns = next((e.text for e in svc if _localname(e.tag) == "Namespace"), "")
                xa = next((e.text for e in svc if _localname(e.tag) == "XAddr"), None)
                if not xa:
                    continue
                if ns and "ptz" in ns:
                    ptz_xaddr = self._rewrite_host(xa)
                elif ns and "media" in ns and media_xaddr is None:
                    media_xaddr = self._rewrite_host(xa)
            if not ptz_xaddr:
                raise RuntimeError("camera does not expose an ONVIF PTZ service")
            if not media_xaddr:
                media_xaddr = f"{self._ep.base}/onvif/media_service"

            profiles = self._post(media_xaddr, f'<GetProfiles xmlns="{_NS_MEDIA}"/>')
            token = None
            for prof in _findall_local(profiles, "Profiles"):
                token = prof.get("token")
                if token:
                    break
            if not token:
                raise RuntimeError("camera returned no ONVIF media profiles")
            self._ptz_xaddr = ptz_xaddr
            self._profile_token = token

    def continuous_move(self, pan: float, tilt: float, zoom: float,
                        timeout_s: float = 2.0) -> None:
        """Start moving at the given normalised velocities (-1..1).

        A short ONVIF Timeout makes the camera auto-stop if the client stops
        sending (e.g. the browser tab closes mid-drag); the UI re-sends while
        the button is held to keep motion smooth.
        """
        self._discover()
        body = (
            f'<ContinuousMove xmlns="{_NS_PTZ}">'
            f"<ProfileToken>{self._profile_token}</ProfileToken>"
            f'<Velocity xmlns:t="{_NS_SCHEMA}">'
            f'<t:PanTilt x="{pan:.3f}" y="{tilt:.3f}"/>'
            f'<t:Zoom x="{zoom:.3f}"/>'
            "</Velocity>"
            f"<Timeout>PT{timeout_s:.1f}S</Timeout>"
            "</ContinuousMove>"
        )
        self._post(self._ptz_xaddr, body)  # type: ignore[arg-type]

    def stop(self) -> None:
        self._discover()
        body = (
            f'<Stop xmlns="{_NS_PTZ}">'
            f"<ProfileToken>{self._profile_token}</ProfileToken>"
            "<PanTilt>true</PanTilt><Zoom>true</Zoom></Stop>"
        )
        self._post(self._ptz_xaddr, body)  # type: ignore[arg-type]

    def home(self) -> None:
        """Recall the camera's configured home preset (GotoHomePosition)."""
        self._discover()
        body = (
            f'<GotoHomePosition xmlns="{_NS_PTZ}">'
            f"<ProfileToken>{self._profile_token}</ProfileToken>"
            "</GotoHomePosition>"
        )
        self._post(self._ptz_xaddr, body)  # type: ignore[arg-type]


class PtzManager:
    """Owns one OnvifPtz per PTZ-enabled camera and routes control requests."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._clients: Dict[str, OnvifPtz] = {}
        self._lock = threading.Lock()

    def ptz_cameras(self) -> List[str]:
        return [c.name for c in self._settings.cameras if c.ptz]

    def _client(self, camera: str) -> OnvifPtz:
        with self._lock:
            client = self._clients.get(camera)
            if client is not None:
                return client
            cfg = self._settings.camera(camera)
            if not cfg.ptz:
                raise KeyError(camera)
            endpoint = _resolve_endpoint(cfg)
            if endpoint is None:
                raise RuntimeError(f"camera {camera} has no ONVIF host configured")
            client = OnvifPtz(endpoint)
            self._clients[camera] = client
            return client

    def move(self, camera: str, pan: float, tilt: float, zoom: float) -> None:
        self._client(camera).continuous_move(pan, tilt, zoom)

    def stop(self, camera: str) -> None:
        self._client(camera).stop()

    def home(self, camera: str) -> None:
        self._client(camera).home()
