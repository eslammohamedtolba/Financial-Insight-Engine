
from passlib.context import CryptContext
from datetime import timedelta, datetime
from app.helpers.settings import settings
import jwt
import logging
import uuid

class BaseController:
    def __init__(self):
        self.passwd_context = CryptContext(
            schemes=['bcrypt']
        )

    def generate_passwd_hash(self, password: str) -> str:
        return self.passwd_context.hash(password)
        
    def verify_passwd(self, password: str, hash: str) -> bool:
        return self.passwd_context.verify(password, hash)

    def create_secret_token(user_data: dict, expiry: timedelta = timedelta(seconds=3600), refresh: bool = False):
        
        token_data = {}
        
        token_data['sub'] = user_data
        token_data['exp'] = datetime.now() + expiry
        token_data['jti'] = str(uuid.uuid4())
        token_data['refresh'] = refresh
        
        token = jwt.encode(
            payload=token_data,
            key=settings.JWT_SECRET,
            algorithm=settings.JWT_ALGORITHM
        )

        return token

    def decode_token(token: str) -> dict:
        try:
            token_data = jwt.decode(
                jwt=token,
                key=settings.JWT_SECRET,
                algorithms=[settings.JWT_ALGORITHM]
            )
            return token_data
        
        except jwt.PyJWKError as e:
            logging.exception(e) 
            return None
