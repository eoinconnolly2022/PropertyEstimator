import base64
from datetime import datetime, timedelta
from random import SystemRandom
import bcrypt
import psycopg2
from psycopg2 import sql

#Database configuration
DB_CONFIG = {
    "dbname": "mydb",
    "user": "postgres",
    "password": "Password",  
    "host": "localhost",
    "port": 5432,
}
# initialize random number generator
cryptogen = SystemRandom()

#helper function to execute queries
def execute_query(query, params=None, fetch=False):
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    try:
        cursor.execute(query, params)
        if fetch:
            result = cursor.fetchall()
        else:
            result = None
        conn.commit()
        return result
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        conn.rollback()
        return None
    finally:
        cursor.close()
        conn.close()

# Create a new user account given a username and password
def create_account(username, password):
    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    #Securely generate an API key for the user
    random_bytes = cryptogen.getrandbits(512).to_bytes(64, 'big')
    api_key = base64.b64encode(random_bytes).decode()

    user_query = sql.SQL("""
    INSERT INTO Users (username, password_hash)
    VALUES (%s, %s)
    RETURNING user_id
    """)
    user_id = execute_query(user_query, (username, hashed_password), fetch=True)

    if user_id:
        
        api_key_query = sql.SQL("""
        INSERT INTO APIKeys (user_id, api_key, created_at, expires_at)
        VALUES (%s, %s, %s, %s)
        """)
        created_at = datetime.now()
        expires_at = created_at + timedelta(days=365)  # API key expiration set for one year in the future
        execute_query(api_key_query, (user_id[0][0], api_key, created_at, expires_at))

        response = {
            "status": "success",
            "timestamp": created_at.isoformat(),
            "requestId": "123",
            "apiKey": {
                "apiKey": api_key,
                "expiresAt": expires_at.isoformat()
            }
        }
    else:
        response = {
            "status": "failure",
            "message": "Failed to create account"
        }
    return response

# Get all API keys for a user given username and password
def get_api_key(username, password):
    query = sql.SQL("""
    SELECT user_id, password_hash FROM Users WHERE username = %s
    """)
    result = execute_query(query, (username,), fetch=True)

    if result:
        user_id, stored_hashed_password = result[0][0], result[0][1].encode()
        if bcrypt.checkpw(password.encode(), stored_hashed_password):
            api_key_query = sql.SQL("""
            SELECT api_key, created_at, expires_at FROM APIKeys
            WHERE user_id = %s
            ORDER BY created_at DESC
            """)
            api_key_result = execute_query(api_key_query, (user_id,), fetch=True)

            if api_key_result:
                api_keys = [
                    {
                        "apiKey": row[0],
                        "createdAt": row[1].isoformat(),
                        "expiresAt": row[2].isoformat()
                    }
                    for row in api_key_result
                ]
                response = {
                    "status": "success",
                    "apiKeys": api_keys
                }
            else:
                response = {
                    "status": "failure",
                    "message": "No API keys found for the user"
                }
        else:
            response = {
                "status": "failure",
                "message": "Invalid username or password"
            }
    else:
        response = {
            "status": "failure",
            "message": "Invalid username or password"
        }
    return response

# Change password using old password and new password
def change_password(username, old_password, new_password):
    query = sql.SQL("""
    SELECT user_id, password_hash FROM Users WHERE username = %s
    """)
    result = execute_query(query, (username,), fetch=True)

    if result:
        user_id, stored_hashed_password = result[0][0], result[0][1].encode()
        if bcrypt.checkpw(old_password.encode(), stored_hashed_password):
            new_hashed_password = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
            update_query = sql.SQL("""
            UPDATE Users SET password_hash = %s WHERE user_id = %s
            """)
            execute_query(update_query, (new_hashed_password, user_id))
            response = {
                "status": "success",
                "message": "Password reset successfully"
            }
        else:
            response = {
                "status": "failure",
                "message": "Invalid username or password"
            }
    else:
        response = {
            "status": "failure",
            "message": "Invalid username or password"
        }
    return response

# Get account info given username and password
def account_info(username, password):
    api_key_response = get_api_key(username, password)
    if api_key_response["status"] == "success":
        response = {
            "status": "success",
            "username": username,
            "apiKeys": api_key_response["apiKeys"],
            "usage": "unlimited"
        }
    else:
        response = {
            "status": "failure",
            "message": "Invalid username or password"
        }
    return response

# Create a new API key for an existing user given username and password
def create_new_api_key(username, password):
    query = sql.SQL("""
    SELECT user_id, password_hash FROM Users WHERE username = %s
    """)
    result = execute_query(query, (username,), fetch=True)

    if result:
        user_id, stored_hashed_password = result[0][0], result[0][1].encode()
        if bcrypt.checkpw(password.encode(), stored_hashed_password):
            #Securely generate a new API key
            random_bytes = cryptogen.getrandbits(512).to_bytes(64, 'big')
            api_key = base64.b64encode(random_bytes).decode()

            api_key_query = sql.SQL("""
            INSERT INTO APIKeys (user_id, api_key, created_at, expires_at)
            VALUES (%s, %s, %s, %s)
            """)
            created_at = datetime.now()
            expires_at = created_at + timedelta(days=365) #Set expiration date to 1 year in the future
            execute_query(api_key_query, (user_id, api_key, created_at, expires_at))

            response = {
                "status": "success",
                "timestamp": created_at.isoformat(),
                "requestId": "123",
                "apiKey": {
                    "apiKey": api_key,
                    "expiresAt": expires_at.isoformat()
                }
            }
        else:
            response = {
                "status": "failure",
                "message": "Invalid username or password"
            }
    else:
        response = {
            "status": "failure",
            "message": "Invalid username or password"
        }
    return response

# Get usage data for a user
def get_user_usage(username, password):
    query = sql.SQL("""
    SELECT user_id, password_hash FROM Users WHERE username = %s
    """)
    result = execute_query(query, (username,), fetch=True)

    if result:
        user_id, stored_hashed_password = result[0][0], result[0][1].encode()
        if bcrypt.checkpw(password.encode(), stored_hashed_password):
            usage_query = sql.SQL("""
            SELECT timestamp, endpoint, request_data, response_data
            FROM UsageData
            WHERE api_key_id IN (
                SELECT api_key_id FROM APIKeys WHERE user_id = %s
            )
            ORDER BY timestamp DESC
            """)
            usage_data = execute_query(usage_query, (user_id,), fetch=True)

            if usage_data:
                response = {
                    "status": "success",
                    "usage": [
                        {
                            "timestamp": row[0].isoformat(),
                            "endpoint": row[1],
                            "request_data": row[2],
                            "response_data": row[3]
                        }
                        for row in usage_data
                    ]
                }
            else:
                response = {
                    "status": "success",
                    "usage": []
                }
        else:
            response = {
                "status": "failure",
                "message": "Invalid username or password"
            }
    else:
        response = {
            "status": "failure",
            "message": "Invalid username or password"
        }
    return response