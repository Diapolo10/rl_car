import json
import hashlib
from pathlib import Path
from PIL import Image

cacheFile = "cache.json"

def getHitboxFromCache(imageFile):
    """Gets hitbox for a given image file from the cache"""
    # Check if cachefile exists
    path = Path(cacheFile)
    if path.is_file() is False:
        # Create cachefile and return None if it doesn't exist
        with open(cacheFile, 'w') as f:
            templatejson = {"key":"data"}
            f.write(json.dumps(templatejson))
            print("No saved hitbox for the given imagefile.")
        return None

    # Read cachefile and load json data from it
    with open(cacheFile, 'r') as f:
        cache = json.loads(f.read())
    # Create hash from given image file
    imageHash = hashlib.md5(Image.open(imageFile).tobytes()).hexdigest()
    # Returns data if key matches with just generated hash.
    try:
        data = cache[imageHash]
        print("Hitbox data fetched from cache.")
        return data
    except:
        print("No saved hitbox for the given imagefile.")
        return None
             


def addHitboxToCache(imageFile, data):
    """Adds hitbox for a image file to the cache"""
    try:
        # Generate a hash from the image file
        imageHash = hashlib.md5(Image.open(imageFile).tobytes()).hexdigest()
        # Create a object with hash and hitbox data
        objectToAdd = { imageHash : data }
        # Write to cache file
        with open(cacheFile, 'r+') as f:
            cache = json.loads(f.read())
            cache.update(objectToAdd)
            f.seek(0)
            f.write(json.dumps(cache))
            f.truncate()
            print("Successfully added generated hitbox to cache.")
    except:
        print("Adding hitbox to cache failed.")
