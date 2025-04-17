// Global variables
let sessionId = generateSessionId();
let currentSection = 'chat';
let currentSudoku = null;
let quizData = null;

// DOM Ready handler
document.addEventListener('DOMContentLoaded', function() {
    // Navigation handlers
    setupNavigation();
    
    // Chat functionality
    document.getElementById('send-btn').addEventListener('click', sendMessage);
    document.getElementById('user-input').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Study resources handlers
    document.getElementById('find-videos-btn').addEventListener('click', findStudyVideos);
    
    // Quiz handlers
    loadQuizTopics();
    document.getElementById('start-quiz-btn').addEventListener('click', startQuiz);
    document.getElementById('submit-answer-btn').addEventListener('click', submitQuizAnswer);
    document.getElementById('next-question-btn').addEventListener('click', loadNextQuizQuestion);
    
    // Brain games handlers
    document.getElementById('start-sudoku-btn').addEventListener('click', startSudoku);
    document.getElementById('check-sudoku-btn').addEventListener('click', checkSudoku);
    document.getElementById('new-sudoku-btn').addEventListener('click', startSudoku);
    document.getElementById('start-memory-btn').addEventListener('click', () => {
        alert('Memory game coming soon!');
    });
    
    // Media suggestions handlers
    document.getElementById('suggest-songs-btn').addEventListener('click', suggestSongs);
    document.getElementById('suggest-videos-btn').addEventListener('click', suggestVideos);
    
    // Emotion analysis handlers
    document.getElementById('analyze-emotion-btn').addEventListener('click', analyzeEmotion);
});

// Navigation setup
function setupNavigation() {
    const navLinks = document.querySelectorAll('nav a');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Remove active class from all links
            navLinks.forEach(l => l.classList.remove('active-nav'));
            
            // Add active class to clicked link
            this.classList.add('active-nav');
            
            // Hide all sections
            document.querySelectorAll('main > div[id$="-section"]').forEach(section => {
                section.classList.add('hidden');
            });
            
            // Show the corresponding section
            const sectionId = this.id.replace('-nav', '-section');
            document.getElementById(sectionId).classList.remove('hidden');
            
            // Update page title
            const sectionName = this.querySelector('span').textContent;
            document.getElementById('page-title').textContent = sectionName;
            
            // Update current section
            currentSection = this.id.replace('-nav', '');
        });
    });
}

// Chat functionality
function sendMessage() {
    const userInput = document.getElementById('user-input');
    const messageText = userInput.value.trim();
    
    if (messageText) {
        // Add user message to chat
        appendMessage('user', messageText);
        
        // Clear input field
        userInput.value = '';
        
        // Show loading indicator
        appendMessage('ai', '<em>Thinking...</em>', 'loading-message');
        
        // Send message to backend
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: messageText,
                session_id: sessionId
            })
        })
        .then(response => response.json())
        .then(data => {
            // Remove loading indicator
            document.querySelector('.loading-message')?.remove();
            
            // Add AI response to chat
            appendMessage('ai', data.response);
            
            // Scroll to bottom
            const chatContainer = document.querySelector('.chat-container');
            chatContainer.scrollTop = chatContainer.scrollHeight;
        })
        .catch(error => {
            console.error('Error:', error);
            document.querySelector('.loading-message')?.remove();
            appendMessage('ai', 'Sorry, I encountered an error. Please try again.');
        });
    }
}

function appendMessage(sender, content, className = '') {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `flex justify-${sender === 'user' ? 'end' : 'start'} ${className}`;
    
    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = `message-bubble ${sender === 'user' ? 'user-bubble' : 'ai-bubble'} p-3 shadow-sm`;
    bubbleDiv.innerHTML = content;
    
    messageDiv.appendChild(bubbleDiv);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    const chatContainer = document.querySelector('.chat-container');
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Study resources functionality
function findStudyVideos() {
    const topic = document.getElementById('study-topic').value.trim();
    const subTopic = document.getElementById('study-subtopic').value.trim();
    
    if (topic) {
        // Show "loading" state
        document.getElementById('video-results').classList.remove('hidden');
        document.getElementById('video-list').innerHTML = '<p>Loading videos...</p>';
        
        fetch('/api/videos', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                topic: topic,
                sub_topic: subTopic
            })
        })
        .then(response => response.json())
        .then(data => {
            const videoList = document.getElementById('video-list');
            videoList.innerHTML = '';
            
            if (data.videos && data.videos.length > 0) {
                data.videos.forEach(videoUrl => {
                    const linkItem = document.createElement('div');
                    linkItem.className = 'p-3 bg-gray-50 rounded-lg';
                    
                    const link = document.createElement('a');
                    link.href = videoUrl;
                    link.target = '_blank';
                    link.className = 'text-indigo-600 hover:text-indigo-800';
                    link.textContent = videoUrl;
                    
                    linkItem.appendChild(link);
                    videoList.appendChild(linkItem);
                });
            } else {
                videoList.innerHTML = '<p>No videos found. Try different search terms.</p>';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('video-list').innerHTML = '<p>Error fetching videos. Please try again.</p>';
        });
    } else {
        alert('Please enter a study topic.');
    }
}

// Quiz functionality
function loadQuizTopics() {
    fetch('/api/quiz/topics')
        .then(response => response.json())
        .then(data => {
            const topicSelect = document.getElementById('quiz-topic');
            
            data.topics.forEach(topic => {
                const option = document.createElement('option');
                option.value = topic.toLowerCase();
                option.textContent = topic;
                topicSelect.appendChild(option);
            });
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

function startQuiz() {
    const topic = document.getElementById('quiz-topic').value;
    
    if (!topic) {
        alert('Please select a topic.');
        return;
    }
    
    // Show loading state
    document.getElementById('quiz-start').classList.add('hidden');
    document.getElementById('quiz-question').classList.remove('hidden');
    document.getElementById('question-text').textContent = 'Loading question...';
    document.getElementById('options-list').innerHTML = '';
    
    // Fetch quiz question
    loadQuizQuestion(topic);
}

function loadQuizQuestion(topic) {
    fetch('/api/quiz', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            topic: topic
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            document.getElementById('quiz-start').classList.remove('hidden');
            document.getElementById('quiz-question').classList.add('hidden');
            return;
        }
        
        quizData = data;
        
        // Update question
        document.getElementById('question-text').textContent = decodeHtmlEntities(data.question);
        
        // Create options
        const optionsList = document.getElementById('options-list');
        optionsList.innerHTML = '';
        
        data.options.forEach((option, index) => {
            const optionDiv = document.createElement('div');
            optionDiv.className = 'flex items-center p-3 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer';
            
            const radio = document.createElement('input');
            radio.type = 'radio';
            radio.name = 'quiz-option';
            radio.id = `option-${index}`;
            radio.value = option;
            radio.className = 'mr-3';
            
            const label = document.createElement('label');
            label.htmlFor = `option-${index}`;
            label.className = 'flex-1 cursor-pointer';
            label.textContent = decodeHtmlEntities(option);
            
            optionDiv.appendChild(radio);
            optionDiv.appendChild(label);
            
            // Make the entire div clickable
            optionDiv.addEventListener('click', () => {
                radio.checked = true;
            });
            
            optionsList.appendChild(optionDiv);
        });
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error loading quiz. Please try again.');
        document.getElementById('quiz-start').classList.remove('hidden');
        document.getElementById('quiz-question').classList.add('hidden');
    });
}

function submitQuizAnswer() {
    const selectedOption = document.querySelector('input[name="quiz-option"]:checked');
    
    if (!selectedOption) {
        alert('Please select an answer.');
        return;
    }
    
    const answer = selectedOption.value;
    
    fetch('/api/verify_answer', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            answer: answer,
            correct_answer: quizData.correct_answer
        })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('quiz-question').classList.add('hidden');
        document.getElementById('quiz-result').classList.remove('hidden');
        
        const resultMessage = document.getElementById('result-message');
        resultMessage.className = `p-4 rounded-lg mb-4 ${data.is_correct ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`;
        
        if (data.is_correct) {
            resultMessage.innerHTML = '<i class="fas fa-check-circle mr-2"></i> Correct! Great job!';
        } else {
            resultMessage.innerHTML = `<i class="fas fa-times-circle mr-2"></i> Incorrect. The correct answer is: <strong>${decodeHtmlEntities(data.correct_answer)}</strong>`;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error verifying answer. Please try again.');
    });
}

function loadNextQuizQuestion() {
    document.getElementById('quiz-result').classList.add('hidden');
    document.getElementById('quiz-question').classList.remove('hidden');
    
    const topic = document.getElementById('quiz-topic').value;
    loadQuizQuestion(topic);
}

// Brain games functionality
function startSudoku() {
    document.getElementById('sudoku-game').classList.remove('hidden');
    document.getElementById('sudoku-board').innerHTML = '<p>Loading Sudoku puzzle...</p>';
    
    fetch('/api/sudoku')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                document.getElementById('sudoku-board').innerHTML = `<p class="text-red-500">${data.error}</p>`;
                return;
            }
            
            currentSudoku = data;
            renderSudokuBoard(data.puzzle);
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('sudoku-board').innerHTML = '<p class="text-red-500">Error loading Sudoku puzzle. Please try again.</p>';
        });
}

function renderSudokuBoard(puzzleData) {
    const board = document.getElementById('sudoku-board');
    board.innerHTML = '';
    
    for (let row = 0; row < 9; row++) {
        const rowDiv = document.createElement('div');
        rowDiv.className = 'flex';
        
        for (let col = 0; col < 9; col++) {
            const cell = document.createElement('input');
            cell.type = 'text';
            cell.className = 'sudoku-cell';
            cell.dataset.row = row;
            cell.dataset.col = col;
            cell.maxLength = 1;
            
            // Apply borders to create 3x3 grid sections
            if (row % 3 === 0) cell.style.borderTop = '2px solid #333';
            if (row === 8) cell.style.borderBottom = '2px solid #333';
            if (col % 3 === 0) cell.style.borderLeft = '2px solid #333';
            if (col === 8) cell.style.borderRight = '2px solid #333';
            
            const value = puzzleData[row][col];
            if (value !== 0) {
                cell.value = value;
                cell.readOnly = true;
                cell.classList.add('sudoku-cell-fixed');
            } else {
                cell.addEventListener('input', function(e) {
                    // Only allow numbers 1-9
                    if (!/^[1-9]$/.test(e.target.value)) {
                        e.target.value = '';
                    }
                });
            }
            
            rowDiv.appendChild(cell);
        }
        
        board.appendChild(rowDiv);
    }
}

function checkSudoku() {
    if (!currentSudoku || !currentSudoku.solution) {
        alert('No active Sudoku puzzle.');
        return;
    }
    
    let isCorrect = true;
    let isComplete = true;
    
    for (let row = 0; row < 9; row++) {
        for (let col = 0; col < 9; col++) {
            const cell = document.querySelector(`.sudoku-cell[data-row="${row}"][data-col="${col}"]`);
            const value = parseInt(cell.value) || 0;
            
            if (value === 0) {
                isComplete = false;
            } else if (value !== currentSudoku.solution[row][col]) {
                isCorrect = false;
                cell.classList.add('bg-red-200');
                setTimeout(() => {
                    cell.classList.remove('bg-red-200');
                }, 1500);
            }
        }
    }
    
    if (!isComplete) {
        alert('The puzzle is not complete yet. Please fill in all cells.');
    } else if (isCorrect) {
        alert('Congratulations! You solved the Sudoku puzzle correctly!');
    } else {
        alert('There are some errors in your solution. Incorrect cells are highlighted.');
    }
}

// Media suggestions functionality
function suggestSongs() {
    const genre = document.getElementById('music-genre').value;
    
    document.getElementById('songs-results').classList.remove('hidden');
    document.getElementById('songs-list').innerHTML = '<p>Finding songs for you...</p>';
    
    fetch('/api/suggest_song', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            session_id: sessionId,
            genre: genre
        })
    })
    .then(response => response.json())
    .then(data => {
        const songsList = document.getElementById('songs-list');
        songsList.innerHTML = '';
        
        document.getElementById('song-emotion').textContent = data.emotion || 'preferences';
        
        if (data.songs && data.songs.length > 0) {
            data.songs.forEach(songUrl => {
                const songItem = document.createElement('div');
                songItem.className = 'p-2 bg-gray-50 rounded';
                
                const songLink = document.createElement('a');
                songLink.href = songUrl;
                songLink.target = '_blank';
                songLink.className = 'text-indigo-600 hover:text-indigo-800';
                
                // Extract song name from URL if possible
                const songName = songUrl.includes('/track/') 
                    ? songUrl.split('/track/')[1].split('?')[0].replace(/-/g, ' ') 
                    : songUrl;
                
                songLink.textContent = songName;
                songItem.appendChild(songLink);
                songsList.appendChild(songItem);
            });
        } else {
            songsList.innerHTML = '<p>No songs found based on your mood.</p>';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('songs-list').innerHTML = '<p>Error suggesting songs. Please try again.</p>';
    });
}

function suggestVideos() {
    document.getElementById('videos-results').classList.remove('hidden');
    document.getElementById('videos-list').innerHTML = '<p>Finding videos for you...</p>';
    
    fetch('/api/suggest_video', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            session_id: sessionId
        })
    })
    .then(response => response.json())
    .then(data => {
        const videosList = document.getElementById('videos-list');
        videosList.innerHTML = '';
        
        document.getElementById('video-emotion').textContent = data.emotion || 'preferences';
        
        if (data.videos && data.videos.length > 0) {
            data.videos.forEach(videoUrl => {
                const videoItem = document.createElement('div');
                videoItem.className = 'p-2 bg-gray-50 rounded';
                
                const videoLink = document.createElement('a');
                videoLink.href = videoUrl;
                videoLink.target = '_blank';
                videoLink.className = 'text-indigo-600 hover:text-indigo-800';
                videoLink.textContent = videoUrl;
                
                videoItem.appendChild(videoLink);
                videosList.appendChild(videoItem);
            });
        } else {
            videosList.innerHTML = '<p>No videos found based on your mood.</p>';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('videos-list').innerHTML = '<p>Error suggesting videos. Please try again.</p>';
    });
}

// Emotion analysis functionality
function analyzeEmotion() {
    const text = document.getElementById('emotion-text').value.trim();
    
    if (!text) {
        alert('Please enter some text to analyze.');
        return;
    }
    
    document.getElementById('emotion-results').classList.remove('hidden');
    document.getElementById('primary-emotion').textContent = 'Analyzing...';
    document.getElementById('emotion-bars').innerHTML = '';
    
    fetch('/api/emotion', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            text: text
        })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('primary-emotion').textContent = data.primary_emotion;
        
        const emotionBars = document.getElementById('emotion-bars');
        emotionBars.innerHTML = '';
        
        // Sort emotions by score
        const emotions = Object.entries(data.emotion_scores).sort((a, b) => b[1] - a[1]);
        
        emotions.forEach(([emotion, score]) => {
            const barContainer = document.createElement('div');
            barContainer.className = 'flex items-center';
            
            const label = document.createElement('div');
            label.className = 'w-20 text-sm text-gray-700';
            label.textContent = emotion;
            
            const barWrapper = document.createElement('div');
            barWrapper.className = 'flex-1 bg-gray-200 rounded-full h-4';
            
            const bar = document.createElement('div');
            const width = Math.round(score * 100);
            bar.className = 'bg-indigo-600 h-4 rounded-full';
            bar.style.width = `${width}%`;
            
            const percentage = document.createElement('span');
            percentage.className = 'ml-2 text-xs text-gray-600';
            percentage.textContent = `${width}%`;
            
            barWrapper.appendChild(bar);
            barContainer.appendChild(label);
            barContainer.appendChild(barWrapper);
            barContainer.appendChild(percentage);
            
            emotionBars.appendChild(barContainer);
        });
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('primary-emotion').textContent = 'Error analyzing text';
    });
}

// Helper functions
function generateSessionId() {
    return 'session_' + Math.random().toString(36).substring(2, 15);
}

function decodeHtmlEntities(text) {
    const textArea = document.createElement('textarea');
    textArea.innerHTML = text;
    return textArea.value;
}
