from sqlmodel import SQLModel

# Schema for the API response when a user logs in.
class Token(SQLModel):
    access_token: str
    token_type: str = "bearer"

# Schema to represent the data we embed inside the JWT.
class TokenData(SQLModel):
    sub: str # The "subject" of the token, which will be our user's ID.
