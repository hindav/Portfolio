// Utility: format markdown text to HTML
function formatText(text) {
  if (!text) return '';

  // Escape HTML to prevent XSS (basic)
  let safeText = text.replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // Bold (**text**)
  safeText = safeText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

  // Lists:
  // 1. Match newline OR start of string, followed by * or - and a space
  safeText = safeText.replace(/(?:^|\n)\s*[\*\-]\s+/g, '<br>• ');

  // 2. Handle "inline" lists after a colon (e.g. "include: * item")
  safeText = safeText.replace(/:\s+[\*\-]\s+/g, ':<br>• ');

  // 3. Handle loose * surrounded by spaces if it looks like a separator (fallback)
  // safeText = safeText.replace(/\s\*\s/g, ' • ');

  // Normalize remaining newlines to <br>
  safeText = safeText.replace(/\n/g, '<br>');

  return safeText;
}

// Utility: add message to chat
function addMessage(text, type) {
  const chat = document.getElementById("aiChat");
  if (!chat) return;

  const div = document.createElement("div");
  div.className = `ai-message ${type}`;

  // Use innerHTML for formatted text, textContent for user input if just raw
  if (type === 'ai-bot') {
    div.innerHTML = formatText(text);

    // Add Retry button if rate limited
    if (text.includes("Rate Limit Reached")) {
      const retryBtn = document.createElement("button");
      retryBtn.className = "retry-btn";
      retryBtn.textContent = "🔄 Retry";
      retryBtn.onclick = () => {
        // Get last user message
        const messages = chat.querySelectorAll('.ai-message.ai-user');
        if (messages.length > 0) {
          const lastQuestion = messages[messages.length - 1].textContent;
          const input = document.getElementById('aiQuestion');
          input.value = lastQuestion;
          submitQuestion();
        }
      };
      div.appendChild(retryBtn);
    }
  } else {
    div.textContent = text;
  }

  chat.appendChild(div);

  // 🔥 Expand chat once messages exist
  chat.classList.add("has-messages");

  chat.scrollTop = chat.scrollHeight;
}

// Submit question to API
async function submitQuestion() {
  const input = document.getElementById('aiQuestion');
  const chat = document.getElementById('aiChat');
  if (!input || !chat) return;

  const question = input.value.trim();
  if (!question) return;

  input.value = '';
  addMessage(question, 'ai-user');
  addMessage('Thinking...', 'ai-bot');

  try {
    const res = await fetch('/proxy-chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    });
    const data = await res.json();

    // Remove thinking
    const last = chat.querySelector('.ai-message:last-child');
    if (last && last.textContent.includes('Thinking')) last.remove();

    if (data.success) {
      addMessage(data.answer, 'ai-bot');
    } else {
      addMessage('Sorry, I couldn\'t generate an answer.', 'ai-bot');
    }
  } catch (err) {
    const last = chat.querySelector('.ai-message:last-child');
    if (last && last.textContent.includes('Thinking')) last.remove();
    addMessage('Error: Unable to connect to AI service.', 'ai-bot');
    console.error(err);
  }
}


// Delegated event handling (SAFE for dynamic HTML)
document.addEventListener("click", (e) => {

  // Send button
  if (e.target.closest("#sendQuestion")) {
    submitQuestion();
  }

  // Suggestion buttons
  if (e.target.matches(".grok-suggestions button")) {
    const input = document.getElementById("aiQuestion");
    if (!input) return;
    input.value = e.target.dataset.question;
    submitQuestion();
  }
});

// Enter key support (delegated)
document.addEventListener("keydown", (e) => {
  if (e.target && e.target.id === "aiQuestion" && e.key === "Enter") {
    submitQuestion();
  }
});








