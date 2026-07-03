"""Persistence utilities using SQLAlchemy (SQLite/Postgres compatible).

Provides simple functions to initialize the DB and persist incoming posts
and drift metrics.
"""
import os
from datetime import datetime
from typing import Optional, Any, Dict
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import json


DB_URL = os.environ.get('DB_URL', 'sqlite:///crisislens.db')

engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith('sqlite') else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Post(Base):
    __tablename__ = 'posts'
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    lat = Column(Float, nullable=True)
    lon = Column(Float, nullable=True)
    label = Column(String(32), nullable=True)
    confidence = Column(Float, nullable=True)
    explanation = Column(Text, nullable=True)
    raw_response = Column(Text, nullable=True)


class DriftMetric(Base):
    __tablename__ = 'drift_metrics'
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    disaster_rate = Column(Float, nullable=True)
    avg_confidence = Column(Float, nullable=True)
    n_samples = Column(Integer, nullable=True)
    reasons = Column(Text, nullable=True)


def init_db():
    Base.metadata.create_all(bind=engine)


def save_post(text: str, label: Optional[str], confidence: Optional[float], explanation: Optional[Dict[Any, Any]] = None, raw_response: Optional[Dict[Any, Any]] = None, lat: Optional[float] = None, lon: Optional[float] = None):
    session = SessionLocal()
    try:
        post = Post(
            text=text,
            timestamp=datetime.utcnow(),
            lat=lat,
            lon=lon,
            label=label,
            confidence=confidence,
            explanation=json.dumps(explanation) if explanation is not None else None,
            raw_response=json.dumps(raw_response) if raw_response is not None else None
        )
        session.add(post)
        session.commit()
        session.refresh(post)
        return post.id
    finally:
        session.close()


def save_drift_metric(disaster_rate: Optional[float], avg_confidence: Optional[float], n_samples: int, reasons: Optional[list] = None):
    session = SessionLocal()
    try:
        dm = DriftMetric(
            timestamp=datetime.utcnow(),
            disaster_rate=disaster_rate,
            avg_confidence=avg_confidence,
            n_samples=n_samples,
            reasons=json.dumps(reasons) if reasons is not None else None
        )
        session.add(dm)
        session.commit()
        session.refresh(dm)
        return dm.id
    finally:
        session.close()
