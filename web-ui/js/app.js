// Configuration
const API_BASE_URL = 'http://localhost:8000/api/v1';

// State
let currentUser = null;
let authToken = null;
let conversations = [];
let currentConversationId = null;
let messages = [];
let contextMenuConversationId = null;

// DOM Elements
const authContainer = document.getElementById('authContainer');
const chatContainer = document.getElementById('chatContainer');
const loginForm = document.getElementById('loginForm');
const registerForm = document.getElementById('registerForm');
const loginFormEl = document.getElementById('loginFormEl');
const registerFormEl = document.getElementById('registerFormEl');
const authError = document.getElementById('authError');
const showRegister = document.getElementById('showRegister');
const showLogin = document.getElementById('showLogin');
const logoutBtn = document.getElementById('logoutBtn');
const userName = document.getElementById('userName');
const userAvatar = document.getElementById('userAvatar');
const newChatBtn = document.getElementById('newChatBtn');
const conversationsList = document.getElementById('conversationsList');
const messagesContainer = document.getElementById('messagesContainer');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const contextMenu = document.getElementById('contextMenu');
const renameModal = document.getElementById('renameModal');
const renameInput = document.getElementById('renameInput');
const confirmRename = document.getElementById('confirmRename');
const cancelRename = document.getElementById('cancelRename');

// Utility Functions
function showError(message) {
    authError.textContent = message;
    authError.style.display = 'block';
}

function hideError() {
    authError.style.display = 'none';
}

function setLoading(button, loading) {
    if (loading) {
        button.disabled = true;
        button.textContent = 'Loading...';
    } else {
        button.disabled = false;
        button.textContent = button.id.includes('login') ? 'Sign In' : 'Create Account';
    }
}

function getAuthHeaders() {
    return {
        'Authorization': `Bearer ${authToken}`,
        'Content-Type': 'application/json'
    };
}

function formatMessage(content) {
    // Handle numbered lists and line breaks
    return content
        .replace(/\n/g, '\n')  // Preserve line breaks
        .replace(/(\d+\.)/g, '\n$1')  // Add line break before numbered items
        .trim();
}

// API Functions
async function apiCall(endpoint, options = {}) {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        ...options,
        headers: {
            'Content-Type': 'application/json',
            ...options.headers
        }
    });
    
    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Network error' }));
        throw new Error(error.detail || 'Request failed');
    }
    
    // Handle 204 No Content responses (like DELETE)
    if (response.status === 204) {
        return null;
    }
    
    return response.json();
}

async function login(email, password) {
    const response = await apiCall('/auth/login', {
        method: 'POST',
        body: JSON.stringify({ email, password })
    });
    
    authToken = response.access_token;
    localStorage.setItem('authToken', authToken);
    return response;
}

async function register(userData) {
    return await apiCall('/auth/register', {
        method: 'POST',
        body: JSON.stringify({
            email: userData.email,
            username: userData.username,
            first_name: userData.firstName,
            second_name: userData.lastName,
            password: userData.password
        })
    });
}

async function createConversation() {
    return await apiCall('/conversations', {
        method: 'POST',
        headers: getAuthHeaders()
    });
}

async function getConversations() {
    return await apiCall('/conversations', {
        headers: getAuthHeaders()
    });
}

async function updateConversationName(conversationId, newName) {
    return await apiCall(`/conversations/${conversationId}`, {
        method: 'PATCH',
        headers: getAuthHeaders(),
        body: JSON.stringify({ name: newName })
    });
}

async function deleteConversationApi(conversationId) {
    return await apiCall(`/conversations/${conversationId}`, {
        method: 'DELETE',
        headers: getAuthHeaders()
    });
}

async function sendMessage(conversationId, message) {
    return await apiCall(`/agent/chat/${conversationId}`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify({ query: message })
    });
}

async function getMessages(conversationId) {
    return await apiCall(`/agent/messages/${conversationId}`, {
        headers: getAuthHeaders()
    });
}

// UI Functions
function showAuth() {
    authContainer.style.display = 'flex';
    chatContainer.style.display = 'none';
}

function showChat() {
    authContainer.style.display = 'none';
    chatContainer.style.display = 'flex';
}

function addMessage(role, content) {
    const messageEl = document.createElement('div');
    messageEl.className = `message ${role}`;
    
    const formattedContent = formatMessage(content);
    
    messageEl.innerHTML = `
        <div class="message-avatar">${role === 'user' ? currentUser?.first_name?.charAt(0) || 'U' : 'AI'}</div>
        <div class="message-content">${formattedContent}</div>
    `;
    
    messagesContainer.appendChild(messageEl);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function showLoading() {
    const loadingEl = document.createElement('div');
    loadingEl.className = 'message assistant loading-message';
    loadingEl.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-content loading">
            <div class="loading-spinner"></div>
            Analyzing...
        </div>
    `;
    messagesContainer.appendChild(loadingEl);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    return loadingEl;
}

function hideLoading(loadingEl) {
    if (loadingEl) {
        loadingEl.remove();
    }
}

function renderConversations() {
    conversationsList.innerHTML = conversations.map(conv => `
        <div class="conversation-item ${conv.id === currentConversationId ? 'active' : ''}" 
             onclick="selectConversation('${conv.id}')">
            <div class="conversation-content">
                <div class="conversation-name">${conv.name}</div>
                <div class="conversation-time">${new Date(conv.created_at).toLocaleDateString()}</div>
            </div>
            <button class="conversation-menu" onclick="showContextMenu(event, '${conv.id}')">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 8c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z"/>
                </svg>
            </button>
        </div>
    `).join('');
}

function showContextMenu(event, conversationId) {
    event.stopPropagation();
    contextMenuConversationId = conversationId;
    
    contextMenu.style.display = 'block';
    contextMenu.style.left = event.pageX + 'px';
    contextMenu.style.top = event.pageY + 'px';
    
    // Adjust position if menu goes off screen
    const rect = contextMenu.getBoundingClientRect();
    const windowWidth = window.innerWidth;
    const windowHeight = window.innerHeight;
    
    if (rect.right > windowWidth) {
        contextMenu.style.left = (event.pageX - rect.width) + 'px';
    }
    
    if (rect.bottom > windowHeight) {
        contextMenu.style.top = (event.pageY - rect.height) + 'px';
    }
}

function hideContextMenu() {
    contextMenu.style.display = 'none';
    contextMenuConversationId = null;
}

function showRenameModal(currentName = '') {
    renameInput.value = currentName;
    renameModal.style.display = 'flex';
    renameInput.focus();
    renameInput.select();
}

function hideRenameModal() {
    renameModal.style.display = 'none';
    renameInput.value = '';
}

async function selectConversation(conversationId) {
    currentConversationId = conversationId;
    renderConversations();
    
    // Clear messages and load conversation history
    messagesContainer.innerHTML = '';
    
    try {
        const conversationMessages = await getMessages(conversationId);
        conversationMessages.forEach(msg => {
            addMessage(msg.role, msg.content);
        });
    } catch (error) {
        console.error('Error loading messages:', error);
    }
}

async function handleRenameConversation() {
    if (!contextMenuConversationId || !renameInput.value.trim()) {
        hideRenameModal();
        return;
    }
    
    const newName = renameInput.value.trim();
    const conversationId = contextMenuConversationId;
    
    try {
        const updatedConversation = await updateConversationName(conversationId, newName);
        
        // Update the conversation in local state immediately
        const index = conversations.findIndex(c => c.id === conversationId);
        if (index !== -1) {
            conversations[index] = updatedConversation;
            renderConversations();
        }
        
        // Close modal and context menu
        hideRenameModal();
        hideContextMenu();
        
        console.log(`Conversation renamed to: ${newName}`);
        
    } catch (error) {
        console.error('Error renaming conversation:', error);
        hideRenameModal();
        hideContextMenu();
        
        const errorMsg = error.message || 'Failed to rename conversation';
        alert(`Rename failed: ${errorMsg}. Please try again.`);
    }
}

async function handleDeleteConversation() {
    if (!contextMenuConversationId) return;
    
    if (!confirm('Are you sure you want to delete this conversation? This action cannot be undone.')) {
        hideContextMenu();
        return;
    }
    
    const conversationToDelete = contextMenuConversationId;
    const wasCurrentConversation = currentConversationId === conversationToDelete;
    
    try {
        // Call the delete API - it returns 204 No Content, so no response to parse
        await deleteConversationApi(conversationToDelete);
        
        // Remove from local state immediately
        conversations = conversations.filter(c => c.id !== conversationToDelete);
        
        // If we deleted the current conversation, clear the chat and reset
        if (wasCurrentConversation) {
            currentConversationId = null;
            messagesContainer.innerHTML = `
                <div class="message assistant">
                    <div class="message-avatar">AI</div>
                    <div class="message-content">
                        Welcome! I'm your Financial Analyst Assistant. I can help you analyze SEC filings for major tech companies including Apple (AAPL), Microsoft (MSFT), Google (GOOG), Amazon (AMZN), and Meta (META). Ask me about risk factors, management discussions, or any specific financial insights you need.
                    </div>
                </div>
            `;
        }
        
        // Update UI immediately
        renderConversations();
        hideContextMenu();
        
        console.log('Conversation deleted successfully');
        
    } catch (error) {
        console.error('Error deleting conversation:', error);
        hideContextMenu();
        
        // Show user-friendly error message
        const errorMsg = error.message || 'Failed to delete conversation';
        alert(`Delete failed: ${errorMsg}. Please try again.`);
    }
}

// Event Handlers
showRegister.addEventListener('click', (e) => {
    e.preventDefault();
    loginForm.style.display = 'none';
    registerForm.style.display = 'block';
    hideError();
});

showLogin.addEventListener('click', (e) => {
    e.preventDefault();
    registerForm.style.display = 'none';
    loginForm.style.display = 'block';
    hideError();
});

loginFormEl.addEventListener('submit', async (e) => {
    e.preventDefault();
    hideError();
    
    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;
    const loginBtn = document.getElementById('loginBtn');
    
    setLoading(loginBtn, true);
    
    try {
        await login(email, password);
        currentUser = { email, first_name: email.split('@')[0] };
        userName.textContent = currentUser.first_name;
        userAvatar.textContent = currentUser.first_name.charAt(0).toUpperCase();
        
        // Load conversations
        conversations = await getConversations();
        renderConversations();
        
        showChat();
    } catch (error) {
        showError(error.message);
    } finally {
        setLoading(loginBtn, false);
    }
});

registerFormEl.addEventListener('submit', async (e) => {
    e.preventDefault();
    hideError();
    
    const userData = {
        email: document.getElementById('registerEmail').value,
        username: document.getElementById('registerUsername').value,
        firstName: document.getElementById('registerFirstName').value,
        lastName: document.getElementById('registerLastName').value,
        password: document.getElementById('registerPassword').value
    };
    
    const registerBtn = document.getElementById('registerBtn');
    setLoading(registerBtn, true);
    
    try {
        await register(userData);
        // Auto-login after registration
        await login(userData.email, userData.password);
        currentUser = userData;
        userName.textContent = userData.firstName;
        userAvatar.textContent = userData.firstName.charAt(0).toUpperCase();
        
        conversations = await getConversations();
        renderConversations();
        
        showChat();
    } catch (error) {
        showError(error.message);
    } finally {
        setLoading(registerBtn, false);
    }
});

logoutBtn.addEventListener('click', () => {
    authToken = null;
    currentUser = null;
    currentConversationId = null;
    conversations = [];
    messages = [];
    localStorage.removeItem('authToken');
    showAuth();
});

newChatBtn.addEventListener('click', async () => {
    try {
        const newConversation = await createConversation();
        conversations.unshift(newConversation);
        renderConversations();
        selectConversation(newConversation.id);
    } catch (error) {
        console.error('Error creating conversation:', error);
    }
});

// Context menu handlers
document.getElementById('renameConversation').addEventListener('click', () => {
    const conversation = conversations.find(c => c.id === contextMenuConversationId);
    showRenameModal(conversation?.name || '');
});

document.getElementById('deleteConversation').addEventListener('click', () => {
    handleDeleteConversation();
});

// Modal handlers
confirmRename.addEventListener('click', handleRenameConversation);
cancelRename.addEventListener('click', hideRenameModal);

// Handle Enter key in rename input
renameInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        handleRenameConversation();
    }
});

// Close modal on escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        hideRenameModal();
        hideContextMenu();
    }
});

// Close context menu when clicking outside - improved version
document.addEventListener('click', (e) => {
    // Don't close if clicking on a conversation menu button
    if (e.target.closest('.conversation-menu')) {
        return;
    }
    
    // Close context menu if clicking outside of it
    if (!contextMenu.contains(e.target)) {
        hideContextMenu();
    }
});

// Close modal when clicking overlay
renameModal.addEventListener('click', (e) => {
    if (e.target === renameModal) {
        hideRenameModal();
    }
});

// Message input handlers
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessageHandler();
    }
});

messageInput.addEventListener('input', () => {
    messageInput.style.height = 'auto';
    messageInput.style.height = messageInput.scrollHeight + 'px';
});

sendBtn.addEventListener('click', sendMessageHandler);

async function sendMessageHandler() {
    const message = messageInput.value.trim();
    if (!message || !currentConversationId) return;
    
    // Add user message to UI
    addMessage('user', message);
    messageInput.value = '';
    messageInput.style.height = 'auto';
    
    // Show loading
    const loadingEl = showLoading();
    sendBtn.disabled = true;
    
    try {
        const response = await sendMessage(currentConversationId, message);
        hideLoading(loadingEl);
        addMessage('assistant', response.response);
        
        // Refresh conversations list to get updated names if auto-generated
        conversations = await getConversations();
        renderConversations();
        
    } catch (error) {
        hideLoading(loadingEl);
        addMessage('assistant', 'I apologize, but I encountered an error processing your request. Please try again.');
        console.error('Error sending message:', error);
    } finally {
        sendBtn.disabled = false;
    }
}

// Initialize app
function init() {
    const savedToken = localStorage.getItem('authToken');
    if (savedToken) {
        authToken = savedToken;
        // Try to validate token by making a request
        getConversations()
            .then(convs => {
                conversations = convs;
                renderConversations();
                currentUser = { first_name: 'User' }; // Simplified for demo
                userName.textContent = 'User';
                userAvatar.textContent = 'U';
                showChat();
            })
            .catch(() => {
                localStorage.removeItem('authToken');
                authToken = null;
                showAuth();
            });
    } else {
        showAuth();
    }
}

// Start the app
init();