"""Simple alerting utilities: email and Slack webhook alerts."""
import os
import smtplib
from email.message import EmailMessage
import requests
import logging

logger = logging.getLogger('crisislens.alert')


def send_email_alert(subject: str, body: str) -> bool:
    """Send an email alert using SMTP settings from env vars.

    Required env vars: ALERT_SMTP_HOST, ALERT_SMTP_PORT, ALERT_FROM, ALERT_TO
    Optional: ALERT_SMTP_USER, ALERT_SMTP_PASS (if auth required)
    """
    host = os.environ.get('ALERT_SMTP_HOST')
    port = int(os.environ.get('ALERT_SMTP_PORT', 25))
    sender = os.environ.get('ALERT_FROM')
    recipient = os.environ.get('ALERT_TO')

    # Treat placeholder/dummy values as disabled (do not attempt to send)
    if not host or not sender or not recipient or host.lower() == 'dummy' or sender.lower() == 'dummy' or recipient.lower() == 'dummy':
        logger.info('Email alerting disabled or configured with dummy credentials; skipping send')
        return False

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = recipient
    msg.set_content(body)

    try:
        user = os.environ.get('ALERT_SMTP_USER')
        password = os.environ.get('ALERT_SMTP_PASS')
        with smtplib.SMTP(host, port, timeout=10) as s:
            s.ehlo()
            if os.environ.get('ALERT_SMTP_TLS', 'false').lower() in ('true','1'):
                s.starttls()
            if user and password:
                s.login(user, password)
            s.send_message(msg)
        return True
    except Exception as e:
        logger.exception('Failed to send email')
        return False


def send_slack_alert(message: str) -> bool:
    """Send a Slack alert to a webhook URL provided in ALERT_SLACK_WEBHOOK env var.

    If webhook is 'DUMMY' or missing, log and skip sending.
    """
    webhook = os.environ.get('ALERT_SLACK_WEBHOOK')
    if not webhook or webhook.lower() == 'dummy':
        logger.info('Slack alerting disabled or configured with dummy webhook; skipping send')
        return False
    try:
        resp = requests.post(webhook, json={'text': message}, timeout=5)
        resp.raise_for_status()
        return True
    except Exception:
        logger.exception('Failed to send Slack alert')
        return False
