"""Twitter/X streaming client that forwards matching tweets to the Phase 5 API.

This script uses Tweepy v4 StreamingClient (Twitter API v2). Provide credentials
via environment variables and a filter rule (or set up custom rules via the
Twitter developer portal).

Env vars expected:
 - TWITTER_BEARER_TOKEN
 - PREDICTOR_URL (e.g. http://localhost:8000/predict)

Run as:
    python deployment/twitter_streamer.py

Note: Ensure `tweepy` is installed and you have a developer account with access to streaming endpoints.
"""
import os
import requests
import json
import logging
from tweepy import StreamingClient, StreamRule

logger = logging.getLogger('crisislens.twitter')


class ForwardingStream(StreamingClient):
    def __init__(self, bearer_token, predictor_url):
        super().__init__(bearer_token)
        self.predictor_url = predictor_url

    def on_connect(self):
        logger.info('Connected to Twitter stream')

    def on_tweet(self, tweet):
        # tweet.text contains the text; you may want to enrich with geo/user
        payload = {'text': tweet.text, 'model_type': 'classical'}
        try:
            r = requests.post(self.predictor_url, json=payload, timeout=10)
            logger.info(f'Forwarded tweet id={tweet.id} status={r.status_code}')
        except Exception as e:
            logger.exception('Failed to forward tweet')


def main():
    bearer = os.environ.get('TWITTER_BEARER_TOKEN')
    predictor = os.environ.get('PREDICTOR_URL', 'http://localhost:8000/predict')

    # If no bearer token provided, run a local simulation fallback
    if not bearer or bearer.lower() == 'dummy':
        logger.info('No Twitter bearer token provided; running local simulation forwarding sample tweets to predictor')
        # Simple loop forwarding sample tweets
        import time
        samples = [
            'Massive earthquake downtown',
            'Wildfire approaching homes, evacuate now',
            'Flooding after heavy rains',
            'Lovely sunny day at the park',
        ]
        try:
            while True:
                for s in samples:
                    payload = {'text': s, 'model_type': 'classical'}
                    try:
                        r = requests.post(predictor, json=payload, timeout=10)
                        logger.info(f'Forwarded sample tweet status={r.status_code}')
                    except Exception:
                        logger.exception('Failed to forward sample tweet')
                    time.sleep(2.0)
        except KeyboardInterrupt:
            logger.info('Simulation stopped by user')
        return

    stream = ForwardingStream(bearer, predictor)

    # Example: add a rule to track disaster-related keywords
    rules = [StreamRule('earthquake OR flood OR fire OR tornado OR hurricane')]
    try:
        # Remove existing rules for a clean start (optional)
        existing_rules = stream.get_rules()
        if existing_rules and existing_rules.data:
            ids = [r.id for r in existing_rules.data]
            stream.delete_rules(ids)
    except Exception:
        pass

    stream.add_rules(rules)
    stream.filter(tweet_fields=['text'])


if __name__ == '__main__':
    main()
