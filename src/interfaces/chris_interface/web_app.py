"""
================================================================================
FUND OPS WEB INTERFACE - Chris Friedman Agent Communication Portal
================================================================================
Author: Tom Hogan
Developer: Alpha Loop Capital, LLC

A modern, professional web interface for Chris Friedman and Tom Hogan to
communicate directly with SANTAS_HELPER and CPA agents.

Features:
- Real-time chat with agents
- Report requests and viewing
- Status dashboards
- Conversation history
- Secure Azure-based deployment

Run with:
    python -m src.interfaces.chris_interface.web_app

================================================================================
"""

import logging
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, session
from flask_cors import CORS
import secrets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
CORS(app)

# Global communicator instance
_communicator = None

def get_communicator():
    """Get or initialize the communicator"""
    global _communicator
    if _communicator is None:
        try:
            from src.interfaces.chris_interface.agent_communicator import get_communicator as gc
            _communicator = gc()
        except Exception as e:
            logger.error(f"Failed to initialize communicator: {e}")
            _communicator = None
    return _communicator


# ===============================================================================
# HTML TEMPLATE - Modern, Professional UI
# ===============================================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Alpha Loop Capital - Fund Operations</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0f1a;
            --bg-secondary: #111827;
            --bg-tertiary: #1f2937;
            --bg-card: #162032;
            --accent-primary: #3b82f6;
            --accent-secondary: #10b981;
            --accent-gold: #f59e0b;
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --text-muted: #6b7280;
            --border-color: #374151;
            --shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            --gradient-blue: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            --gradient-green: linear-gradient(135deg, #10b981 0%, #059669 100%);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow: hidden;
        }

        /* Background Pattern */
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background:
                radial-gradient(circle at 20% 20%, rgba(59, 130, 246, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(16, 185, 129, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 50% 50%, rgba(245, 158, 11, 0.03) 0%, transparent 50%);
            pointer-events: none;
            z-index: 0;
        }

        .app-container {
            display: flex;
            height: 100vh;
            position: relative;
            z-index: 1;
        }

        /* Sidebar */
        .sidebar {
            width: 280px;
            background: var(--bg-secondary);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            padding: 24px 16px;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 0 8px 24px;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 24px;
        }

        .logo-icon {
            width: 40px;
            height: 40px;
            background: var(--gradient-blue);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 18px;
        }

        .logo-text {
            font-weight: 600;
            font-size: 16px;
            line-height: 1.2;
        }

        .logo-text span {
            display: block;
            font-size: 11px;
            font-weight: 400;
            color: var(--text-secondary);
        }

        .nav-section {
            margin-bottom: 24px;
        }

        .nav-section-title {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            padding: 0 8px;
            margin-bottom: 8px;
        }

        .nav-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
            margin-bottom: 4px;
        }

        .nav-item:hover {
            background: var(--bg-tertiary);
        }

        .nav-item.active {
            background: rgba(59, 130, 246, 0.15);
            color: var(--accent-primary);
        }

        .nav-icon {
            width: 20px;
            height: 20px;
            opacity: 0.7;
        }

        .agent-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 16px;
            background: var(--bg-tertiary);
            border-radius: 12px;
            margin-top: auto;
        }

        .agent-status {
            width: 8px;
            height: 8px;
            background: var(--accent-secondary);
            border-radius: 50%;
            animation: pulse 2s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        /* Main Content */
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        /* Header */
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 20px 32px;
            border-bottom: 1px solid var(--border-color);
            background: var(--bg-secondary);
        }

        .header-title {
            font-size: 20px;
            font-weight: 600;
        }

        .header-title span {
            color: var(--accent-primary);
        }

        .user-info {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .user-avatar {
            width: 36px;
            height: 36px;
            background: var(--gradient-gold);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 14px;
        }

        /* Chat Area */
        .chat-area {
            flex: 1;
            overflow-y: auto;
            padding: 24px 32px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .message {
            max-width: 75%;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            align-self: flex-end;
        }

        .message.agent {
            align-self: flex-start;
        }

        .message-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }

        .message-sender {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-secondary);
        }

        .message-time {
            font-size: 11px;
            color: var(--text-muted);
        }

        .message-content {
            padding: 16px 20px;
            border-radius: 16px;
            line-height: 1.6;
        }

        .message.user .message-content {
            background: var(--gradient-blue);
            border-bottom-right-radius: 4px;
        }

        .message.agent .message-content {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-bottom-left-radius: 4px;
        }

        .message.system .message-content {
            background: transparent;
            border: 1px solid var(--border-color);
            border-radius: 12px;
            text-align: center;
            color: var(--text-secondary);
            font-size: 14px;
        }

        .message-content strong {
            color: var(--accent-primary);
        }

        .message-content ul {
            margin: 12px 0;
            padding-left: 20px;
        }

        .message-content li {
            margin-bottom: 6px;
        }

        /* Input Area */
        .input-area {
            padding: 20px 32px;
            border-top: 1px solid var(--border-color);
            background: var(--bg-secondary);
        }

        .input-container {
            display: flex;
            gap: 12px;
            align-items: flex-end;
        }

        .input-wrapper {
            flex: 1;
            position: relative;
        }

        #messageInput {
            width: 100%;
            padding: 16px 20px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            color: var(--text-primary);
            font-family: 'Outfit', sans-serif;
            font-size: 15px;
            resize: none;
            outline: none;
            transition: border-color 0.2s ease;
        }

        #messageInput:focus {
            border-color: var(--accent-primary);
        }

        #messageInput::placeholder {
            color: var(--text-muted);
        }

        .send-btn {
            width: 52px;
            height: 52px;
            background: var(--gradient-blue);
            border: none;
            border-radius: 12px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .send-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        }

        .send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .send-btn svg {
            width: 20px;
            height: 20px;
            fill: white;
        }

        /* Quick Actions */
        .quick-actions {
            display: flex;
            gap: 8px;
            margin-top: 12px;
            flex-wrap: wrap;
        }

        .quick-action {
            padding: 8px 16px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            font-size: 13px;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .quick-action:hover {
            background: var(--bg-card);
            color: var(--text-primary);
            border-color: var(--accent-primary);
        }

        /* Status Panel */
        .status-panel {
            width: 320px;
            background: var(--bg-secondary);
            border-left: 1px solid var(--border-color);
            padding: 24px;
            overflow-y: auto;
        }

        .panel-title {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .status-card {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
        }

        .status-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }

        .status-card-title {
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
        }

        .status-badge {
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 600;
        }

        .status-badge.green {
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-secondary);
        }

        .status-badge.blue {
            background: rgba(59, 130, 246, 0.15);
            color: var(--accent-primary);
        }

        .status-value {
            font-size: 24px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            margin-bottom: 4px;
        }

        .status-label {
            font-size: 12px;
            color: var(--text-muted);
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
        }

        ::-webkit-scrollbar-track {
            background: transparent;
        }

        ::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 3px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-muted);
        }
    </style>
</head>
<body>
    <div class="app-container">
        <!-- Sidebar -->
        <aside class="sidebar">
            <div class="logo">
                <div class="logo-icon">AL</div>
                <div class="logo-text">
                    Alpha Loop Capital
                    <span>Fund Operations Portal</span>
                </div>
            </div>

            <nav class="nav-section">
                <div class="nav-section-title">Communication</div>
                <div class="nav-item active">
                    <svg class="nav-icon" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/>
                    </svg>
                    <span>Chat with Agents</span>
                </div>
            </nav>

            <nav class="nav-section">
                <div class="nav-section-title">Reports</div>
                <div class="nav-item" onclick="quickMessage('Give me the daily NAV status')">
                    <svg class="nav-icon" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z"/>
                    </svg>
                    <span>NAV Status</span>
                </div>
                <div class="nav-item" onclick="quickMessage('What is the K-1 preparation status?')">
                    <svg class="nav-icon" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zM6 20V4h7v5h5v11H6z"/>
                    </svg>
                    <span>Tax Status</span>
                </div>
                <div class="nav-item" onclick="quickMessage('Audit status update please')">
                    <svg class="nav-icon" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M19 3h-4.18C14.4 1.84 13.3 1 12 1c-1.3 0-2.4.84-2.82 2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-7 0c.55 0 1 .45 1 1s-.45 1-1 1-1-.45-1-1 .45-1 1-1z"/>
                    </svg>
                    <span>Audit Status</span>
                </div>
            </nav>

            <div class="agent-indicator">
                <div class="agent-status"></div>
                <div>
                    <div style="font-weight: 600; font-size: 13px;">Agents Online</div>
                    <div style="font-size: 11px; color: var(--text-muted);">SANTAS_HELPER • CPA</div>
                </div>
            </div>
        </aside>

        <!-- Main Content -->
        <main class="main-content">
            <header class="header">
                <h1 class="header-title">Fund Operations <span>Assistant</span></h1>
                <div class="user-info">
                    <span id="userName">Chris Friedman</span>
                    <div class="user-avatar">CF</div>
                </div>
            </header>

            <div class="chat-area" id="chatArea">
                <!-- Messages will be inserted here -->
            </div>

            <div class="input-area">
                <div class="input-container">
                    <div class="input-wrapper">
                        <textarea
                            id="messageInput"
                            placeholder="Ask SANTAS_HELPER or CPA anything..."
                            rows="1"
                            onkeydown="handleKeyDown(event)"
                        ></textarea>
                    </div>
                    <button class="send-btn" onclick="sendMessage()" id="sendBtn">
                        <svg viewBox="0 0 24 24">
                            <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                        </svg>
                    </button>
                </div>
                <div class="quick-actions">
                    <span class="quick-action" onclick="quickMessage('Daily status update')">Daily Status</span>
                    <span class="quick-action" onclick="quickMessage('Current NAV?')">NAV</span>
                    <span class="quick-action" onclick="quickMessage('Upcoming deadlines')">Deadlines</span>
                    <span class="quick-action" onclick="quickMessage('LP report status')">LP Reports</span>
                </div>
            </div>
        </main>

        <!-- Status Panel -->
        <aside class="status-panel">
            <div class="panel-title">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M3 13h2v-2H3v2zm0 4h2v-2H3v2zm0-8h2V7H3v2zm4 4h14v-2H7v2zm0 4h14v-2H7v2zM7 7v2h14V7H7z"/>
                </svg>
                Quick Status
            </div>

            <div class="status-card">
                <div class="status-card-header">
                    <span class="status-card-title">Fund NAV</span>
                    <span class="status-badge green">Final</span>
                </div>
                <div class="status-value">$127.4M</div>
                <div class="status-label">As of Dec 31, 2024</div>
            </div>

            <div class="status-card">
                <div class="status-card-header">
                    <span class="status-card-title">K-1 Progress</span>
                    <span class="status-badge blue">On Track</span>
                </div>
                <div class="status-value">80%</div>
                <div class="status-label">40 of 50 investors</div>
            </div>

            <div class="status-card">
                <div class="status-card-header">
                    <span class="status-card-title">MTD Return</span>
                    <span class="status-badge green">+2.3%</span>
                </div>
                <div class="status-value">+18.7%</div>
                <div class="status-label">YTD Performance</div>
            </div>

            <div class="status-card">
                <div class="status-card-header">
                    <span class="status-card-title">Next Deadline</span>
                </div>
                <div class="status-value" style="font-size: 18px;">Feb 29</div>
                <div class="status-label">Form PF Q4 Filing</div>
            </div>
        </aside>
    </div>

    <script>
        let conversationId = null;

        // Initialize conversation on page load
        window.addEventListener('load', async () => {
            await startConversation();
        });

        async function startConversation() {
            try {
                const response = await fetch('/api/start_conversation', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user_id: 'CF' })
                });
                const data = await response.json();
                conversationId = data.conversation_id;

                // Show welcome message
                if (data.messages && data.messages.length > 0) {
                    data.messages.forEach(msg => addMessage(msg));
                }
            } catch (error) {
                console.error('Failed to start conversation:', error);
                addSystemMessage('Connection issue. Please refresh.');
            }
        }

        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const content = input.value.trim();

            if (!content || !conversationId) return;

            // Disable input while sending
            input.disabled = true;
            document.getElementById('sendBtn').disabled = true;

            // Add user message immediately
            addMessage({
                sender: 'CF',
                type: 'user_message',
                content: content,
                timestamp: new Date().toISOString()
            });

            input.value = '';

            try {
                const response = await fetch('/api/send_message', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        conversation_id: conversationId,
                        content: content
                    })
                });
                const data = await response.json();

                if (data.agent_message) {
                    addMessage(data.agent_message);
                }
            } catch (error) {
                console.error('Failed to send message:', error);
                addSystemMessage('Failed to send message. Please try again.');
            } finally {
                input.disabled = false;
                document.getElementById('sendBtn').disabled = false;
                input.focus();
            }
        }

        function addMessage(msg) {
            const chatArea = document.getElementById('chatArea');
            const messageDiv = document.createElement('div');

            const isUser = msg.sender === 'CF' || msg.sender === 'TJH';
            const isSystem = msg.type === 'system_message';

            messageDiv.className = `message ${isSystem ? 'system' : (isUser ? 'user' : 'agent')}`;

            const time = new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            const sender = isUser ? 'You' : (isSystem ? 'System' : msg.sender);

            let contentHtml = msg.content
                .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
                .replace(/\\n/g, '<br>')
                .replace(/• /g, '<li>')
                .replace(/<li>(.+?)(<br>|$)/g, '<li>$1</li>');

            if (contentHtml.includes('<li>')) {
                contentHtml = contentHtml.replace(/(<li>.*<\\/li>)+/g, '<ul>$&</ul>');
            }

            messageDiv.innerHTML = `
                <div class="message-header">
                    <span class="message-sender">${sender}</span>
                    <span class="message-time">${time}</span>
                </div>
                <div class="message-content">${contentHtml}</div>
            `;

            chatArea.appendChild(messageDiv);
            chatArea.scrollTop = chatArea.scrollHeight;
        }

        function addSystemMessage(content) {
            addMessage({
                sender: 'SYSTEM',
                type: 'system_message',
                content: content,
                timestamp: new Date().toISOString()
            });
        }

        function quickMessage(content) {
            document.getElementById('messageInput').value = content;
            sendMessage();
        }

        function handleKeyDown(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }

        // Auto-resize textarea
        document.getElementById('messageInput').addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 150) + 'px';
        });
    </script>
</body>
</html>
'''


# ===============================================================================
# API Routes
# ===============================================================================

@app.route('/')
def index():
    """Serve the main page"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/start_conversation', methods=['POST'])
def start_conversation():
    """Start a new conversation"""
    data = request.get_json()
    user_id = data.get('user_id', 'CF')

    communicator = get_communicator()
    if communicator is None:
        return jsonify({
            "error": "Agents not available",
            "conversation_id": "offline_" + datetime.now().strftime('%Y%m%d_%H%M%S'),
            "messages": [{
                "sender": "SYSTEM",
                "type": "system_message",
                "content": "Fund operations agents are being initialized. Some features may be limited.",
                "timestamp": datetime.now().isoformat()
            }]
        })

    try:
        conv = communicator.start_conversation(user_id)
        session['conversation_id'] = conv.conversation_id

        return jsonify({
            "conversation_id": conv.conversation_id,
            "messages": [msg.to_dict() for msg in conv.messages]
        })
    except Exception as e:
        logger.error(f"Failed to start conversation: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/send_message', methods=['POST'])
def send_message():
    """Send a message and get response"""
    data = request.get_json()
    conversation_id = data.get('conversation_id')
    content = data.get('content')

    if not conversation_id or not content:
        return jsonify({"error": "Missing conversation_id or content"}), 400

    communicator = get_communicator()
    if communicator is None:
        # Return mock response when agents aren't available
        return jsonify({
            "user_message": {
                "sender": "CF",
                "type": "user_message",
                "content": content,
                "timestamp": datetime.now().isoformat()
            },
            "agent_message": {
                "sender": "SANTAS_HELPER",
                "type": "agent_response",
                "content": "**SANTAS_HELPER**\\n\\nI'm currently initializing. Please try again in a moment.",
                "timestamp": datetime.now().isoformat()
            }
        })

    try:
        user_msg, agent_msg = communicator.send_message(conversation_id, content)

        return jsonify({
            "user_message": user_msg.to_dict(),
            "agent_message": agent_msg.to_dict()
        })
    except Exception as e:
        logger.error(f"Failed to send message: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/get_history', methods=['GET'])
def get_history():
    """Get conversation history"""
    conversation_id = request.args.get('conversation_id')
    limit = int(request.args.get('limit', 50))

    communicator = get_communicator()
    if communicator is None:
        return jsonify({"messages": []})

    history = communicator.get_conversation_history(conversation_id, limit)
    return jsonify({"messages": history})


@app.route('/api/request_report', methods=['POST'])
def request_report():
    """Request a specific report"""
    data = request.get_json()
    conversation_id = data.get('conversation_id')
    report_type = data.get('report_type')
    parameters = data.get('parameters', {})

    communicator = get_communicator()
    if communicator is None:
        return jsonify({"error": "Agents not available"}), 503

    try:
        report_msg = communicator.request_report(conversation_id, report_type, parameters)
        return jsonify({"report": report_msg.to_dict()})
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agents_available": get_communicator() is not None
    })


# ===============================================================================
# Main Entry Point
# ===============================================================================

def main():
    """Run the web application"""
    import argparse

    parser = argparse.ArgumentParser(description="Fund Operations Web Interface")
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    print(f"""
================================================================================
ALPHA LOOP CAPITAL - Fund Operations Interface
================================================================================
Starting server at http://{args.host}:{args.port}

This interface allows Chris Friedman and Tom Hogan to communicate directly
with SANTAS_HELPER and CPA agents.

Features:
  • Real-time chat with fund operations agents
  • Report requests and generation
  • Status dashboards
  • Conversation history with training data collection

Press Ctrl+C to stop the server.
================================================================================
""")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()

