# VoiceCare Backend

A FastAPI-based backend service for VoiceCare - a real-time voice translation system designed for healthcare communication between patients and nurses who speak different languages.

## Features

- **Real-time Voice Processing**: Speech-to-text, translation, and text-to-speech
- **WebSocket Communication**: Real-time messaging and audio streaming
- **Multi-provider Support**: OpenAI, ElevenLabs, and LibreTranslate integration
- **User Management**: Profile creation and authentication
- **Conversation History**: Persistent chat storage and retrieval
- **Health Monitoring**: Performance metrics and health checks

## Tech Stack

- **Framework**: FastAPI 0.104.1
- **Language**: Python 3.8+
- **Database**: SQLAlchemy with SQLite/PostgreSQL support
- **WebSocket**: Native FastAPI WebSocket support
- **Audio Processing**: faster-whisper, ffmpeg-python
- **Translation**: LibreTranslate, OpenAI GPT-4
- **TTS/STT**: OpenAI Whisper, ElevenLabs, OpenAI TTS

## Prerequisites

- Python 3.8 or higher
- pip or poetry
- FFmpeg (for audio processing)
- Git

## Installation

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

4. **Environment Configuration**
   Create a `.env` file in the backend directory:
   ```env
   # Database
   DATABASE_URL=sqlite+aiosqlite:///./voicecare.db
   
   # Server
   APP_HOST=127.0.0.1
   APP_PORT=8000
   
   # CORS
   CORS_ORIGINS=["http://localhost:3000"]
   
   # API Keys (optional)
   OPENAI_API_KEY=your_openai_api_key_here
   ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
   
   # Service Configuration
   STT_PROVIDER=elevenlabs  # options: elevenlabs, openai, whisper
   TTS_PROVIDER=elevenlabs  # options: elevenlabs, openai, browser
   TRANSLATION_PROVIDER=auto  # options: auto, openai, libre
   
   # Voice Settings
   MAX_VOICE_SECONDS=120
   WHISPER_MODEL=tiny
   ```

5. **Initialize Database**
   ```bash
   python -c "from app.db.seed import init_database; import asyncio; asyncio.run(init_database())"
   ```

6. **Start the server**
   ```bash
   python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```

## Quick Start with Script

Use the provided startup script:

```bash
chmod +x start_backend.sh
./start_backend.sh
```

## API Documentation

Once the server is running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## Project Structure

```
├── app/
│   ├── api/v1/           # API routes
│   │   ├── routes_auth.py
│   │   ├── routes_capabilities.py
│   │   ├── routes_conversations.py
│   │   ├── routes_messages.py
│   │   ├── routes_stt.py
│   │   ├── routes_tts.py
│   │   ├── routes_users.py
│   │   └── ws.py
│   ├── core/             # Core configuration
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── security.py
│   ├── db/               # Database layer
│   │   ├── models.py
│   │   ├── schemas.py
│   │   ├── crud.py
│   │   └── session.py
│   ├── services/         # Business logic
│   │   ├── stt_*.py
│   │   ├── tts_*.py
│   │   └── translate_*.py
│   ├── workers/          # Background tasks
│   └── main.py
├── alembic/              # Database migrations
├── requirements.txt
└── start_backend.sh
```

## API Endpoints

### Authentication
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/refresh` - Refresh token

### Users
- `GET /api/v1/users/` - List users
- `POST /api/v1/users/` - Create user
- `GET /api/v1/users/{user_id}` - Get user details
- `PUT /api/v1/users/{user_id}` - Update user
- `DELETE /api/v1/users/{user_id}` - Delete user

### Conversations
- `GET /api/v1/conversations/` - List conversations
- `POST /api/v1/conversations/` - Create conversation
- `GET /api/v1/conversations/{conversation_id}` - Get conversation

### Messages
- `GET /api/v1/messages/` - List messages
- `POST /api/v1/messages/` - Create message
- `GET /api/v1/messages/{message_id}` - Get message

### Voice Processing
- `POST /api/v1/stt/transcribe` - Speech-to-text
- `POST /api/v1/tts/synthesize` - Text-to-speech
- `GET /api/v1/capabilities/` - Get service capabilities

### WebSocket
- `WS /api/v1/ws` - Real-time communication

## Service Providers

### Speech-to-Text (STT)
- **ElevenLabs**: High-quality, fast transcription
- **OpenAI Whisper**: Reliable, open-source option
- **Local Whisper**: Offline processing with faster-whisper

### Text-to-Speech (TTS)
- **ElevenLabs**: High-quality, natural voices
- **OpenAI TTS**: Good quality, multiple voices
- **Browser TTS**: Client-side synthesis

### Translation
- **LibreTranslate**: Free, open-source translation
- **OpenAI GPT-4**: High-quality, context-aware translation

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Database connection string | `sqlite+aiosqlite:///./voicecare.db` |
| `APP_HOST` | Server host | `127.0.0.1` |
| `APP_PORT` | Server port | `8000` |
| `CORS_ORIGINS` | Allowed CORS origins | `["http://localhost:3000"]` |
| `STT_PROVIDER` | Speech-to-text provider | `elevenlabs` |
| `TTS_PROVIDER` | Text-to-speech provider | `elevenlabs` |
| `TRANSLATION_PROVIDER` | Translation provider | `auto` |
| `OPENAI_API_KEY` | OpenAI API key | `None` |
| `ELEVENLABS_API_KEY` | ElevenLabs API key | `None` |
| `MAX_VOICE_SECONDS` | Maximum recording duration | `120` |

### Database Configuration

The application supports both SQLite and PostgreSQL:

**SQLite (Default)**
```env
DATABASE_URL=sqlite+aiosqlite:///./voicecare.db
```

**PostgreSQL**
```env
DATABASE_URL=postgresql+asyncpg://user:password@localhost/voicecare
```

## Development

### Running Tests
```bash
pytest
```

### Database Migrations
```bash
# Create migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head
```

### Code Quality
```bash
# Format code
black app/

# Lint code
flake8 app/

# Type checking
mypy app/
```

## Deployment

### Docker (Recommended)

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.11-slim
   
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   EXPOSE 8000
   
   CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **Build and run**
   ```bash
   docker build -t voicecare-backend .
   docker run -p 8000:8000 voicecare-backend
   ```

### Production Deployment

1. **Install production dependencies**
   ```bash
   pip install gunicorn
   ```

2. **Run with Gunicorn**
   ```bash
   gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```

3. **Configure reverse proxy** (Nginx example)
   ```nginx
   location / {
       proxy_pass http://127.0.0.1:8000;
       proxy_http_version 1.1;
       proxy_set_header Upgrade $http_upgrade;
       proxy_set_header Connection "upgrade";
       proxy_set_header Host $host;
   }
   ```

## Monitoring

### Health Checks
- `GET /health` - Basic health check
- `GET /api/v1/metrics` - Performance metrics
- `GET /debug/connections` - WebSocket connection status

### Logging
Logs are configured in `app/core/logging.py` with different levels:
- INFO: General application flow
- WARNING: Non-critical issues
- ERROR: Errors that don't stop the application
- CRITICAL: Fatal errors

## Troubleshooting

### Common Issues

1. **Database connection failed**
   - Check DATABASE_URL format
   - Ensure database server is running
   - Verify credentials

2. **Audio processing errors**
   - Install FFmpeg: `apt-get install ffmpeg` (Ubuntu) or `brew install ffmpeg` (macOS)
   - Check audio file formats
   - Verify service provider API keys

3. **WebSocket connection issues**
   - Check CORS configuration
   - Verify WebSocket URL
   - Check firewall settings

4. **Translation not working**
   - Verify internet connection
   - Check API keys for paid services
   - Try different translation providers

### Debug Mode

Enable debug logging:
```env
LOG_LEVEL=DEBUG
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For technical support:
1. Check the troubleshooting section
2. Review logs for error details
3. Open an issue on GitHub
4. Contact the development team

---

**VoiceCare Backend** - Powering healthcare communication through technology.
