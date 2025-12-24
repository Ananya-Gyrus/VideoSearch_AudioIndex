import os
import json
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
import time

DATABASE_URL = ""
WORKING_DIR = ""

def initialize_db_config(app):
    global DATABASE_URL, WORKING_DIR
    DATABASE_URL = app.config.get('DATABASE_URL', '')
    WORKING_DIR = app.config.get('WORKING_DIR', 'work_dir')


Base = declarative_base()

class VideoMetadata(Base):
    __tablename__ = 'video_metadata'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    faiss_id = Column(Integer, index=True)
    source_id = Column(String(255), index=True)
    video_filename = Column(String(500))
    video_path_relative = Column(Text)
    total_scenes = Column(Integer)
    scene_index = Column(Integer)  # For video scenes
    chunk_index_start = Column(Integer)  # For text chunks
    chunk_index_end = Column(Integer)  # For text chunks
    sentence_index = Column(Integer)  # For text sentences
    start_frame = Column(Integer)
    end_frame = Column(Integer)
    start_time_sec = Column(Float)
    end_time_sec = Column(Float)
    duration_sec = Column(Float)
    embedding_filename = Column(String(500))
    embedding_type = Column(String(50))  # 'video' or 'text'
    text_content = Column(Text)  # For storing transcribed text
    no_speech_prob = Column(Float)  # Probability of no speech for audio segments
    database_name = Column(String(255), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Create composite unique constraint for faiss_id + database_name
    __table_args__ = (
        {'mysql_engine': 'InnoDB'}
    )

class DatabaseManager:
    def __init__(self, database_url=None):
        self.database_url = database_url or DATABASE_URL
        self.engine = None
        self.SessionLocal = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database connection and create tables"""
        try:
            self.engine = create_engine(self.database_url, echo=False)
            Base.metadata.create_all(bind=self.engine)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            print("Database initialized successfully")
        except Exception as e:
            print(f"Error initializing database: {e}")
            # Fallback to SQLite for development
            sqlite_url = f"sqlite:///{WORKING_DIR}/video_search.db"
            print(f"Falling back to SQLite: {sqlite_url}")
            self.engine = create_engine(sqlite_url, echo=False)
            Base.metadata.create_all(bind=self.engine)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def insert_metadata_batch(self, metadata_list, database_name):
        """Insert a batch of metadata records"""
        session = self.get_session()
        try:
            # Create all VideoMetadata objects first
            db_metadata_list = []
            for metadata in metadata_list:
                # Common fields for both video and text
                metadata_obj = VideoMetadata(
                    faiss_id=metadata.get('faiss_id'),
                    source_id=metadata.get('source_id'),
                    video_filename=metadata.get('video_filename'),
                    video_path_relative=metadata.get('video_path_relative'),
                    start_frame=metadata.get('start_frame'),
                    end_frame=metadata.get('end_frame'),
                    start_time_sec=metadata.get('start_time_sec'),
                    end_time_sec=metadata.get('end_time_sec'),
                    embedding_filename=metadata.get('embedding_filename'),
                    embedding_type=metadata.get('embedding_type', 'video'),
                    database_name=database_name
                )
                
                # Type-specific fields
                if metadata.get('embedding_type') == 'text':
                    metadata_obj.chunk_index_start = metadata.get('chunk_index_start')
                    metadata_obj.chunk_index_end = metadata.get('chunk_index_end')
                    metadata_obj.sentence_index = metadata.get('sentence_index')
                    metadata_obj.text_content = metadata.get('text')
                    metadata_obj.no_speech_prob = metadata.get('no_speech_prob')
                else:  # video type
                    metadata_obj.scene_index = metadata.get('scene_index')
                    metadata_obj.duration_sec = metadata.get('duration_sec')
                    metadata_obj.total_scenes = metadata.get('total_scenes')

                db_metadata_list.append(metadata_obj)

            session.add_all(db_metadata_list)
            session.commit()
            # print(f"Successfully inserted {len(metadata_list)} metadata records")
        except Exception as e:
            session.rollback()
            print(f"Error inserting metadata batch: {e}")
            raise
        finally:
            session.close()
    
    # def get_metadata_by_database(self, database_name, index_type=None):
    #     """
    #     Get all metadata for a specific database
    #     Args:
    #         database_name: The name of the database to query
    #         index_type: Optional filter for embedding type ('video' or 'text')
    #     Returns:
    #         List of metadata dictionaries
    #     """
    #     session = self.get_session()
    #     try:
    #         query = session.query(VideoMetadata).filter_by(database_name=database_name)
    #         if index_type in ['video', 'text']:
    #             query = query.filter_by(embedding_type=index_type)
    #         metadata_records = query.all()
    #         return [self._metadata_to_dict(record) for record in metadata_records]
    #     except Exception as e:
    #         print(f"Error retrieving metadata: {e}")
    #         return []
    #     finally:
    #         session.close()

    def get_metadata_by_database_dict(self, database_name, index_type=None):
        """
        Get all metadata for a specific database as a dictionary keyed by faiss_id
        Args:
            database_name: The name of the database to query
            index_type: Optional filter for embedding type ('video' or 'text')
        Returns:
            Dictionary of metadata dictionaries keyed by faiss_id
        """
        session = self.get_session()
        try:
            query = session.query(VideoMetadata).filter_by(database_name=database_name)
            if index_type in ['video', 'text']:
                query = query.filter_by(embedding_type=index_type)
            metadata_records = query.all()
            return {record.faiss_id: self._metadata_to_dict(record) for record in metadata_records}
        except Exception as e:
            print(f"Error retrieving metadata: {e}")
            return {}
        finally:
            session.close()
    
    # def get_metadata_by_source_id(self, source_id, database_name=None):
    #     """Get metadata for a specific source_id"""
    #     session = self.get_session()
    #     try:
    #         query = session.query(VideoMetadata).filter_by(source_id=source_id)
    #         if database_name:
    #             query = query.filter_by(database_name=database_name)
    #         metadata_records = query.all()
    #         return [self._metadata_to_dict(record) for record in metadata_records]
    #     except Exception as e:
    #         print(f"Error retrieving metadata by source_id: {e}")
    #         return []
    #     finally:
    #         session.close()
    def get_faiss_ids_by_source_id_and_type(self, source_id, index_type=None ,database_name=None):

        """Get faiss_ids for a specific source_id"""
        session = self.get_session()
        try:
            query = session.query(VideoMetadata.faiss_id).filter_by(source_id=source_id)
            if database_name:
                query = query.filter_by(database_name=database_name)
            if index_type in ['video', 'text']:
                query = query.filter_by(embedding_type=index_type)
            faiss_ids = query.all()
            return [faiss_id[0] for faiss_id in faiss_ids]
        except Exception as e:
            print(f"Error retrieving faiss_ids by source_id: {e}")
            return []
        finally:
            session.close()

    def get_faiss_ids_of_index(self, database_name, index_type=None):
        """Get all faiss_ids for a specific database and embedding type"""
        session = self.get_session()
        try:
            query = session.query(VideoMetadata.faiss_id).filter_by(database_name=database_name)
            if index_type in ['video', 'text']:
                query = query.filter_by(embedding_type=index_type)
            faiss_ids = query.all()
            return [faiss_id[0] for faiss_id in faiss_ids]
        except Exception as e:
            print(f"Error retrieving faiss_ids of index: {e}")
            return []
        finally:
            session.close()

    def get_metadata_by_source_id_and_type(self, source_id, index_type=None ,database_name=None):
        """Get metadata for a specific source_id"""
        session = self.get_session()
        try:
            query = session.query(VideoMetadata).filter_by(source_id=source_id)
            if database_name:
                query = query.filter_by(database_name=database_name)
            if index_type in ['video', 'text']:
                query = query.filter_by(embedding_type=index_type)
            metadata_records = query.all()
            return [self._metadata_to_dict(record) for record in metadata_records]
        except Exception as e:
            print(f"Error retrieving metadata by source_id: {e}")
            return []
        finally:
            session.close()
    
    # def get_metadata_by_faiss_id(self, faiss_id, database_name):
    #     """Get metadata for a specific faiss_id"""
    #     session = self.get_session()
    #     try:
    #         metadata_record = session.query(VideoMetadata).filter_by(
    #             faiss_id=int(faiss_id), database_name=database_name
    #         ).first()
    #         return self._metadata_to_dict(metadata_record) if metadata_record else None
    #     except Exception as e:
    #         print(f"Error retrieving metadata by faiss_id: {e}")
    #         return None
    #     finally:
    #         session.close()
    
    def get_video_stats(self, database_name=None):
        """Get statistics about indexed videos"""
        session = self.get_session()
        try:
            query = session.query(VideoMetadata)
            if database_name:
                query = query.filter_by(database_name=database_name)
            
            # Get total number of scenes (all records)
            total_scenes = query.count()
            
            # Get count of unique videos using source_id
            unique_videos = session.query(VideoMetadata.source_id).\
                filter_by(database_name=database_name) if database_name else session.query(VideoMetadata.source_id)
            unique_videos = unique_videos.distinct().count()
            
            return {
                'total_scenes': total_scenes,
                'unique_files': unique_videos
            }
        except Exception as e:
            print(f"Error getting video stats: {e}")
            return {'total_scenes': 0, 'unique_files': 0}
        finally:
            session.close()
    
    def get_all_databases(self):
        """Get list of all database names"""
        session = self.get_session()
        try:
            databases = session.query(VideoMetadata.database_name).distinct().all()
            return list(set(db[0] for db in databases if db[0]))
        except Exception as e:
            print(f"Error getting database list: {e}")
            return []
        finally:
            session.close()

    def get_indexed_files_by_db_and_type(self):
        session = self.get_session()
        try:
            results = {}
            # Get all metadata for this database
            query = session.query(
                VideoMetadata.database_name,
                VideoMetadata.embedding_type,
                VideoMetadata.source_id,
                func.count(VideoMetadata.id).label('video_count'),
                VideoMetadata.total_scenes
            ).group_by(VideoMetadata.database_name, VideoMetadata.embedding_type, VideoMetadata.source_id, VideoMetadata.total_scenes).all()
            
            for record in query:
                db, emb_type, source_id, video_count, total_scenes = record
                if db not in results:
                    results[db] = {'video': [], 'text': [], 'partial': []}
                if emb_type == "video":
                    # if video_count == total_scenes:
                    #     results[db]["video"].append(source_id)
                    # else:
                    #     results[db]["partial"].append(source_id)
                    results[db]["video"].append(source_id)
                elif emb_type == "text":
                    results[db]["text"].append(source_id)
            for db in results:
                if not results[db]["video"]:
                    del results[db]["video"]
                if not results[db]["text"]:
                    del results[db]["text"]
                if not results[db]["partial"]:
                    del results[db]["partial"]
            return results

        except Exception as e:
            print(f"Error getting indexed files by type: {e}")
            return {}
        finally:
            session.close()

    def get_max_chunk_indexed(self, source_id, database_name):
        """Get the maximum chunk index for a specific source_id"""
        session = self.get_session()
        try:
            max_chunk = session.query(func.max(VideoMetadata.chunk_index_end)).filter_by(
                source_id=source_id,
                database_name=database_name
            ).scalar()
            return max_chunk if max_chunk is not None else -1
        except Exception as e:
            print(f"Error retrieving max chunk index: {e}")
            return -1
        finally:
            session.close()
                
    def remove_metadata_by_source_id_and_type(self, source_id, database_name, embedding_type=None):
        """
        Remove metadata for a specific source_id and embedding type (video/text)
        Args:
            source_id: The source ID to remove metadata for
            database_name: The database name to remove from
            embedding_type: The type of embeddings to remove ('video' or 'text')
        Returns:
            The number of records removed
        """
        session = self.get_session()
        try:
            query = session.query(VideoMetadata).filter_by(
                source_id=source_id,
            )
            if embedding_type:
                query = query.filter_by(embedding_type=embedding_type)
            if database_name:
                query = query.filter_by(database_name=database_name)
            
            removed_count = query.count()
            query.delete()
            session.commit()
            return removed_count
        except Exception as e:
            session.rollback()
            print(f"Error removing metadata by type: {e}")
            return 0
        finally:
            session.close()
            
    def _metadata_to_dict(self, metadata_record):
        """Convert metadata record to dictionary"""
        if not metadata_record:
            return None
        
        metadata_dict = {
            'faiss_id': metadata_record.faiss_id,
            'source_id': metadata_record.source_id,
            'video_filename': metadata_record.video_filename,
            'video_path_relative': metadata_record.video_path_relative,
            'total_scenes': metadata_record.total_scenes,
            'start_frame': metadata_record.start_frame,
            'end_frame': metadata_record.end_frame,
            'start_time_sec': metadata_record.start_time_sec,
            'end_time_sec': metadata_record.end_time_sec,
            'embedding_filename': metadata_record.embedding_filename,
            'embedding_type': metadata_record.embedding_type,
            'database': metadata_record.database_name
        }
        
        # Add type-specific fields
        if metadata_record.embedding_type == 'video':
            metadata_dict.update({
                'scene_index': metadata_record.scene_index,
                'duration_sec': metadata_record.duration_sec
            })
        elif metadata_record.embedding_type == 'text':
            metadata_dict.update({
                'chunk_index_start': metadata_record.chunk_index_start,
                'chunk_index_end': metadata_record.chunk_index_end,
                'sentence_index': metadata_record.sentence_index,
                'text': metadata_record.text_content,
                'no_speech_prob': metadata_record.no_speech_prob
            })
            
        return metadata_dict

# Global database manager instance
db_manager = None

def get_db_manager(db_url=None):
    """Get or create database manager instance"""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager(db_url)
    return db_manager