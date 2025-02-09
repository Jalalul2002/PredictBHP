from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Dataperencanaan(db.Model):
    __tablename__ = 'data_perencanaans'
    id = db.Column(db.Integer, primary_key=True)
    nama_perencanaan = db.Column(db.String(255), nullable=False)
    prodi = db.Column(db.String(255))
    type = db.Column(db.String(255))
    status = db.Column(db.String(255))
    created_by = db.Column(db.Integer)
    updated_by = db.Column(db.Integer)
    created_at = db.Column(db.Date)
    updated_at = db.Column(db.Date)

class Perencanaan(db.Model):
    __tablename__ = 'perencanaans'
    id = db.Column(db.Integer, primary_key=True)
    rencana_id = db.Column(db.Integer, db.ForeignKey('data_perencanaans.id'))
    product_code = db.Column(db.String(255))
    stock = db.Column(db.Integer)
    jumlah_kebutuhan = db.Column(db.Integer)
    # Relasi
    dataperencanaan = db.relationship('Dataperencanaan', backref='perencanaan')
    created_by = db.Column(db.Integer)
    updated_by = db.Column(db.Integer)
    created_at = db.Column(db.Date)
    updated_at = db.Column(db.Date)

class Prediksi(db.Model):
    __tablename__ = 'data_prediksis'
    id = db.Column(db.Integer, primary_key=True)
    tahun_perencanaan = db.Column(db.String(255))
    product_code = db.Column(db.String(255))
    location=db.Column(db.String(255))
    kebutuhan = db.Column(db.Integer)
    created_at = db.Column(db.Date)
    updated_at = db.Column(db.Date)

class Assetlab(db.Model):
    __tablename__ = 'assetlabs'
    product_code = db.Column(db.String(255), primary_key=True)
    stock = db.Column(db.Integer)