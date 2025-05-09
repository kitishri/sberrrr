import redis

def get_redis_connection():

    return redis.Redis(host='redis_flags', port=6379, db=1, decode_responses=True)