import smtplib
from app import send_email

class DummySMTP:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.started = False
        self.logged_in = False
        self.sent = False
    def starttls(self):
        self.started = True
    def login(self, user, pw):
        self.logged_in = True
    def sendmail(self, from_addr, to_addrs, msg):
        self.sent = True
    def quit(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        self.quit()


def test_send_email_success(monkeypatch):
    inst = {}
    def factory(host, port):
        smtp = DummySMTP(host, port)
        inst['smtp'] = smtp
        return smtp
    monkeypatch.setattr(smtplib, 'SMTP', factory)
    settings = {
        'smtp_host': 'smtp.test',
        'smtp_port': 25,
        'smtp_username': 'user',
        'smtp_password': 'pw',
        'smtp_tls': True,
        'smtp_from': 'noreply@test.com',
    }
    assert send_email('dest@test.com', 'Hi', '<p>hi</p>', settings)
    assert inst['smtp'].sent


def test_send_email_no_host():
    settings = {}
    assert not send_email('a@b.com', 'sub', '<p>x</p>', settings)
