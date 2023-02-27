from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey
# from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import create_engine
from sqlalchemy.orm import relationship

from analysis import settings

# Base = declarative_base()
class Base(DeclarativeBase):
    pass


class Metadata(Base):
    __tablename__ = 'metadata'

    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(String, nullable=False, unique=True)
    file_path = Column(String, nullable=False)
    pi = Column(String)
    experiment_number = Column(String)
    dataset_name = Column(String)
    animal_id = Column(String)
    stain = Column(String)
    n_channels = Column(Integer)
    colors = Column(String)
    orientation = Column(String)
    voxel_spacing = Column(String)
    voxel_spacing_units = Column(String)
    treatment = Column(String)
    full_brain = Column(Boolean, default=True)
    modality = Column(String)
    microscope = Column(String)
    organism_type = Column(String)
    technique = Column(String)
    method = Column(String)
    grant_number = Column(String)
    doi = Column(String)
    time_point = Column(Integer)
    route = Column(String)


class Cell(Base):
    __tablename__ = 'cell'

    uuid = Column(String, ForeignKey('metadata.uuid'), primary_key=True)
    time_point = Column(Integer, primary_key=True)
    channel = Column(Integer, primary_key=True)
    z_raw = Column(Float)
    y_raw = Column(Float)
    x_raw = Column(Float)
    raw_coord_units = Column(String)
    z_raw_px = Column(Integer)
    y_raw_px = Column(Integer)
    x_raw_px = Column(Integer)
    is_cell = Column(Boolean)
    type = Column(String)
    atlas_name = Column(String)
    atlas_resolution = Column(String)
    z_downsampled = Column(Integer)
    y_downsampled = Column(Integer)
    x_downsampled = Column(Integer)
    z_transformed = Column(Float)
    y_transformed = Column(Float)
    x_transformed = Column(Float)
    transformed_coord_units = Column(String)
    z_transformed_px = Column(Integer)
    y_transformed_px = Column(Integer)
    x_transformed_px = Column(Integer)
    atlas_structure_name = Column(String)
    atlas_structure_acronym = Column(String)
    atlas_structure_number = Column(Integer)
    metadata_id = Column(Integer, ForeignKey('metadata.id'))


def metadata_table_exists():
    from sqlalchemy import create_engine, MetaData
    database_location = get_db_location_sqlalchemy()
    engine = create_engine(database_location)
    metadata = MetaData()

    # Check if the metadata table exists
    con = engine.connect()
    return engine.dialect.has_table(con, 'metadata')


def metadata_record_exists(file_path):
    from sqlalchemy.orm.exc import NoResultFound
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    if not metadata_table_exists():
        print("Metadata table doesn't exist -> record doesn't exist")
        return False

    db_location = get_db_location_sqlalchemy()
    engine = create_engine(db_location)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        existing_record = session.query(Metadata).filter(Metadata.file_path == file_path).one()
    except NoResultFound:
        return False
    else:
        return True


def get_db_location_sqlalchemy():
    if settings.DB_TYPE == 'sqlite3':
        location = f'sqlite:///{settings.DB_LOCATION}'
    elif settings.DB_TYPE == 'mysql':
        location = f'mysql+pymysql://root:password@localhost/{settings.MYSQL_DB_NAME}'
    else:
        location = None
    return location


def get_db_connection():
    database_url = get_db_location_sqlalchemy()
    engine = create_engine(database_url)
    conn = engine.raw_connection()
    return conn


def create_metadata_table():
    from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String

    database_location = get_db_location_sqlalchemy()
    engine = create_engine(database_location)
    metadata = MetaData()

    # Check if the metadata table exists
    con = engine.connect()
    if engine.dialect.has_table(con, 'metadata'):
        print("Table 'metadata' already exists")
        return

    # Create the metadata table if it doesn't exist
    metadata_table = Table('metadata', metadata,
        Column('id', Integer, primary_key=True),
        Column('uuid', String(50), unique=True, nullable=False),
        Column('file_path', String(200), nullable=False),
        Column('pi', String(50)),
        Column('experiment_number', String(50)),
        Column('dataset_name', String(50)),
        Column('animal_id', String(50)),
        Column('stain', String(50)),
        Column('n_channels', Integer),
        Column('colors', String(50)),
        Column('orientation', String(50)),
        Column('voxel_spacing', String(50)),
        Column('voxel_spacing_units', String(50)),
        Column('treatment', String(50)),
        Column('full_brain', Integer, default=1),
        Column('modality', String(50)),
        Column('microscope', String(50)),
        Column('organism_type', String(50)),
        Column('technique', String(50)),
        Column('method', String(50)),
        Column('grant_number', String(50)),
        Column('doi', String(50)),
        Column('time_point', Integer),
        Column('route', String(50))
    )

    metadata.create_all(engine)



def save_metadata_to_db(ims_files):
    # TODO: for each imaris file, check whether record exists already
    # handle changing paths
    # create the record, create uuid, fill file path
    # save metadata contained in the imaris file to the db
    for file in ims_files:
        con = get_db_connection()
        cur = con.cursor()
        records = cur.execute(f"SELECT * FROM metadata WHERE file_path LIKE '%{os.path.basename(file)}%'").fetchall()
        if not records:
            create_metadata_record(file)
        else:
            # TODO analyze records
            pass



def create_cell_table():
    from sqlalchemy import create_engine, Table, Column, Integer, Text, Float, Boolean, MetaData

    database_location = get_db_location_sqlalchemy()
    engine = create_engine(database_location)
    metadata = MetaData()

    # Define the "cell" table schema
    cell_table = Table(
        'cell',
        metadata,
        Column('uuid', Text),
        Column('time_point', Integer),
        Column('channel', Integer),
        Column('z_raw', Float),
        Column('y_raw', Float),
        Column('x_raw', Float),
        Column('raw_coord_units', Text),
        Column('z_raw_px', Integer),
        Column('y_raw_px', Integer),
        Column('x_raw_px', Integer),
        Column('is_cell', Boolean),
        Column('type', Text),
        Column('atlas_name', Text),
        Column('atlas_resolution', Text),
        Column('z_downsampled', Integer),
        Column('y_downsampled', Integer),
        Column('x_downsampled', Integer),
        Column('z_transformed', Float),
        Column('y_transformed', Float),
        Column('x_transformed', Float),
        Column('transformed_coord_units', Text),
        Column('z_transformed_px', Integer),
        Column('y_transformed_px', Integer),
        Column('x_transformed_px', Integer),
        Column('atlas_structure_name', Text),
        Column('atlas_structure_acronym', Text),
        Column('atlas_structure_number', Integer),
        Column('metadata', Integer)
    )

    # Create the "cell" table
    conn = engine.connect()
    if engine.dialect.has_table(conn, 'cell'):
        print("Table 'cell' already exists")
        return

    cell_table.create(bind=conn, checkfirst=True)
    conn.commit()
    conn.close()

    print("Table 'cell' created")


# def insert_metadata(database_url, metadata_values):
#     from sqlalchemy import create_engine, MetaData, Table, insert
#
#     engine = create_engine(database_url)
#     metadata = MetaData()
#
#     # Reflect the metadata table
#     metadata.reflect(bind=engine, only=['metadata'])
#     metadata_table = metadata.tables['metadata']
#
#     # Insert the metadata values
#     conn = engine.connect()
#     conn.execute(insert(metadata_table).values(metadata_values))
#     conn.commit()
#     conn.close()
#     print("Created metadata record")


def create_metadata_record(ims_file):
    import ulid
    from imaris_ims_file_reader import ims

    # con = get_db_connection()
    # cur = con.cursor()
    # # records = cur.execute(f"SELECT * FROM metadata WHERE file_path LIKE '%{os.path.basename(ims_file)}%'").fetchall()
    # records = cur.execute(f"SELECT * FROM metadata WHERE file_path LIKE '%{os.path.basename(ims_file)}%'") # TODO
    # if records:
    #     print("Record already exists")
    #     return
    #
    # uuid = ulid.new()
    # f = ims(ims_file)
    # n_channels = f.Channels
    # voxel_spacing = f.metaData[0, 0, 0, 'resolution']
    # voxel_spacing_units = 'um'
    # db_location = get_db_location_sqlalchemy()
    # metadata_values = {
    #     "uuid": str(uuid),
    #     "file_path": ims_file,
    #     "n_channels": n_channels,
    #     "voxel_spacing": str(voxel_spacing),
    #     "voxel_spacing_units": voxel_spacing_units
    # }
    # insert_metadata(db_location, metadata_values)

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    # create an engine and session
    db_location = get_db_location_sqlalchemy()
    engine = create_engine(db_location)
    Session = sessionmaker(bind=engine)
    session = Session()

    uuid = ulid.new()
    f = ims(ims_file)
    n_channels = f.Channels
    voxel_spacing = f.metaData[0, 0, 0, 'resolution']
    voxel_spacing_units = 'um'
    metadata_values = {
        "uuid": str(uuid),
        "file_path": ims_file,
        "n_channels": n_channels,
        "voxel_spacing": str(voxel_spacing),
        "voxel_spacing_units": voxel_spacing_units
    }
    new_metadata = Metadata(**metadata_values)
    session.add(new_metadata)
    session.commit()
