document.getElementById('chat-form').addEventListener('submit', async function(event) {
    event.preventDefault();
    
    const userInput = document.getElementById('user-message');
    const message = userInput.value.trim();
    
    if (!message) return;
    
    // Afficher le message de l'utilisateur
    addMessageToChat(message, 'user');
    userInput.value = '';
    
    const loadingDiv = document.getElementById('loading');
    loadingDiv.classList.remove('hidden');
    
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        });
        
        if (!response.ok) {
            throw new Error(`Erreur HTTP: ${response.status}`);
        }
        
        const data = await response.json();
        loadingDiv.classList.add('hidden');
        
        if (data.success) {
            addMessageToChat(data.response, 'bot');
            
            // Si la réponse suggère de créer un plan d'entraînement, afficher le formulaire
            if (data.response.includes("Pour créer un plan d'entraînement personnalisé")) {
                // Vous pourriez ajouter ici du code pour afficher un formulaire
                // ou rediriger vers une page de formulaire
            }
        } else {
            addMessageToChat("Désolé, une erreur s'est produite: " + (data.error || "Erreur inconnue"), 'bot-error');
        }
    } catch (error) {
        loadingDiv.classList.add('hidden');
        addMessageToChat("Problème de connexion au serveur. Veuillez réessayer plus tard.", 'bot-error');
        console.error("Erreur:", error);
    }
});

function addMessageToChat(message, sender) {
    const chatContainer = document.getElementById('chat-container');
    
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', `${sender}-message`);
    
    if (sender === 'user') {
        messageDiv.classList.add('bg-gray-100', 'p-3', 'rounded-lg', 'ml-8', 'text-right');
        messageDiv.innerHTML = `<p>${message}</p>`;
    } else {
        messageDiv.classList.add('bg-blue-50', 'p-3', 'rounded-lg', 'mr-8');
        messageDiv.innerHTML = formatBotResponse(message);
    }
    
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function formatBotResponse(content) {
    // Formater les titres
    content = content.replace(/###\s+(.*?)(?=\n|$)/g, '<h3 class="font-bold text-lg mt-2 mb-1 text-blue-600">$1</h3>');
    
    // Formater les listes
    content = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    content = content.replace(/\*(.*?)\*/g, '<em>$1</em>');
    content = content.replace(/-\s(.*?)(?=\n|$)/g, '<li>$1</li>');
    content = content.replace(/<li>.*?<\/li>/g, function(match) {
        return '<ul class="list-disc pl-5 my-1">' + match + '</ul>';
    });
    
    // Convertir les sauts de ligne en paragraphes
    content = content.replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>');
    content = '<p>' + content + '</p>';
    
    return content;
}
