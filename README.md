# VoiceCare Backend - Complete Voice Translation API Service

A comprehensive FastAPI backend service for real-time voice translation and communication. VoiceCare Backend provides robust APIs for speech-to-text, translation, text-to-speech, and real-time WebSocket communication.

## 🌟 Features

### Core API Services
- **Speech-to-Text (STT)**: Multiple provider support (OpenAI Whisper, ElevenLabs, Mock)
- **Text-to-Speech (TTS)**: Advanced TTS with gender-aware voice selection
- **Translation Services**: Multi-language translation with LibreTranslate and OpenAI
- **Real-time WebSocket**: Bidirectional communication for live voice translation
- **Audio Processing**: Optimized audio handling and streaming

### Database & Data Management
- **SQLAlchemy ORM**: Robust database abstraction layer
- **Alembic Migrations**: Database schema versioning and management
- **User Management**: Complete user authentication and authorization
- **Conversation Storage**: Persistent conversation and message storage
- **CRUD Operations**: Full Create, Read, Update, Delete functionality

### Authentication & Security
- **JWT Authentication**: Secure token-based authentication
- **Password Hashing**: Bcrypt password security
- **User Registration/Login**: Complete auth flow
- **CORS Configuration**: Cross-origin resource sharing setup
- **Input Validation**: Pydantic model validation

### Advanced Features
- **Multi-provider Architecture**: Pluggable service providers
- **Async Processing**: High-performance async/await patterns
- **Error Handling**: Comprehensive error management
- **Logging**: Structured logging with configurable levels
- **Health Monitoring**: System health and metrics endpoints
- **Background Workers**: Async task processing

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip or poetry
- PostgreSQL (recommended) or SQLite
- FFmpeg (for audio processing)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mrafi-ux/lango-voice-chat-backend.git
   cd lango-voice-chat-backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Setup**
   ```bash
   cp .env.example .env
   ```
   
   Configure your environment variables:
   ```env
   # Database
   DATABASE_URL=postgresql://user:password@localhost/voicecare
   
   # API Keys
   OPENAI_API_KEY=your_openai_key
   ELEVENLABS_API_KEY=your_elevenlabs_key
   
   # Server Configuration
   HOST=0.0.0.0
   PORT=8000
   DEBUG=True
   
   # Security
   SECRET_KEY=your_secret_key
   ALGORITHM=HS256
   ACCESS_TOKEN_EXPIRE_MINUTES=30
   ```

5. **Database Setup**
   ```bash
   # Initialize database
   alembic upgrade head
   
   # Seed initial data (optional)
   python -m app.db.seed
   ```

6. **Start the server**
   ```bash
   # Development
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   
   # Production
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

## 📁 Project Structure

```
app/
├── api/v1/              # API route definitions
│   ├── routes_auth.py   # Authentication endpoints
│   ├── routes_users.py  # User management
│   ├── routes_stt.py    # Speech-to-text endpoints
│   ├── routes_tts.py    # Text-to-speech endpoints
│   ├── routes_conversations.py  # Conversation management
│   ├── routes_messages.py       # Message handling
│   ├── routes_capabilities.py   # System capabilities
│   └── ws.py            # WebSocket endpoints
├── core/                # Core configuration
│   ├── config.py        # Application configuration
│   ├── security.py      # Authentication & security
│   └── logging.py       # Logging configuration
├── db/                  # Database layer
│   ├── models.py        # SQLAlchemy models
│   ├── schemas.py       # Pydantic schemas
│   ├── crud.py          # Database operations
│   ├── session.py       # Database session management
│   └── seed.py          # Database seeding
├── services/            # Business logic services
│   ├── stt_*.py         # Speech-to-text providers
│   ├── tts_*.py         # Text-to-speech providers
│   ├── translate_*.py   # Translation services
│   └── metrics.py       # System metrics
├── workers/             # Background workers
│   └── persist.py       # Data persistence workers
└── main.py              # FastAPI application entry point
```

## 🔌 API Endpoints

### Authentication
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/refresh` - Token refresh
- `GET /api/v1/auth/me` - Get current user

### Speech Services
- `POST /api/v1/stt/transcribe` - Speech-to-text conversion
- `POST /api/v1/tts/synthesize` - Text-to-speech synthesis
- `GET /api/v1/capabilities` - Get available services

### Conversation Management
- `GET /api/v1/conversations` - List user conversations
- `POST /api/v1/conversations` - Create new conversation
- `GET /api/v1/conversations/{id}` - Get conversation details
- `DELETE /api/v1/conversations/{id}` - Delete conversation

### WebSocket
- `WS /ws` - Real-time voice translation
- `WS /ws/{conversation_id}` - Conversation-specific WebSocket

## 🗄️ Database Models

### Core Models
- **User**: User accounts and authentication
- **Conversation**: Chat conversations
- **Message**: Individual messages in conversations
- **Capability**: System capabilities and configurations

### Relationships
- User → Conversations (one-to-many)
- Conversation → Messages (one-to-many)
- User → Messages (one-to-many)

## 🔧 Configuration

### Environment Variables
- `DATABASE_URL`: Database connection string
- `SECRET_KEY`: JWT secret key
- `OPENAI_API_KEY`: OpenAI API key
- `ELEVENLABS_API_KEY`: ElevenLabs API key
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `DEBUG`: Debug mode (default: False)

### Service Providers
- **STT Providers**: OpenAI Whisper, ElevenLabs, Mock
- **TTS Providers**: OpenAI, ElevenLabs
- **Translation**: LibreTranslate, OpenAI

## 🚀 Deployment

### Docker Deployment
```bash
# Build image
docker build -t voicecare-backend .

# Run container
docker run -p 8000:8000 voicecare-backend
```

### Production Setup
```bash
# Install production dependencies
pip install -r requirements.txt

# Run with Gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Environment Configuration
Ensure all environment variables are properly configured for your production environment.

## 🧪 Testing

### API Testing
```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_auth.py
```

### Manual Testing
- Use the `/docs` endpoint for interactive API documentation
- Test WebSocket connections with appropriate clients
- Verify database migrations and seeding

## 📊 Monitoring & Logging

### Health Checks
- `GET /health` - Basic health check
- `GET /metrics` - System metrics
- `GET /api/v1/capabilities` - Service availability

### Logging
- Structured JSON logging
- Configurable log levels
- Request/response logging
- Error tracking and reporting

## 🔒 Security Features

- JWT token authentication
- Password hashing with bcrypt
- Input validation and sanitization
- CORS configuration
- Rate limiting (configurable)
- SQL injection prevention

## 🌐 Supported Languages

The backend supports 100+ languages including:
- English, Spanish, French, German, Italian
- Chinese (Simplified/Traditional), Japanese, Korean
- Arabic, Hindi, Portuguese, Russian
- And many more...

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the API documentation at `/docs`
- Review the troubleshooting guide

## 🔄 Version History

- **v1.0.0**: Initial release with core functionality
- **v1.1.0**: Added WebSocket support and real-time features
- **v1.2.0**: Enhanced authentication and user management
- **v1.3.0**: Added multi-provider architecture and optimization

---

**VoiceCare Backend** - Powering multilingual voice communication.
