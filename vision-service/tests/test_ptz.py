"""ONVIF WS-Security header must XML-escape the username so a credential
containing &, <, >, or " can't break the SOAP envelope or inject XML."""

import xml.etree.ElementTree as ET

from app.ptz import _security_header


def test_security_header_escapes_username_special_chars():
    header = _security_header('a&b<c>"d', 'pw')
    # The raw special chars must not appear unescaped in the serialized header.
    assert '<Username>a&b<c>"d</Username>' not in header
    assert '&amp;' in header
    # And the header (wrapped in a root to supply namespaces) must parse as XML.
    ns = 'xmlns:s="http://www.w3.org/2003/05/soap-envelope"'
    root = ET.fromstring(f'<s:Envelope {ns}>{header}</s:Envelope>')
    user_el = next(el for el in root.iter() if el.tag.rsplit('}', 1)[-1] == 'Username')
    assert user_el.text == 'a&b<c>"d'


def test_security_header_plain_username_roundtrips():
    header = _security_header('admin', 'pw')
    assert '<Username>admin</Username>' in header
