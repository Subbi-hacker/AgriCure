<<<<<<< HEAD
from crop import db, app

with app.app_context():
    db.create_all()
=======
from app import db, app

with app.app_context():
    db.create_all()
>>>>>>> 102f86fab59c8be60a1f25ba7a540b8787acc933
