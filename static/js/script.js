document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const resultCard = document.getElementById('result-card');
    const resultLabel = document.getElementById('result-label');
    const resultConfidence = document.getElementById('result-confidence');
    const stateRisk = document.getElementById('state-risk');
    const probSection = document.getElementById('probability-section');
    const probNormal = document.getElementById('prob-normal');
    const probAttack = document.getElementById('prob-attack');
    const probNormalValue = document.getElementById('prob-normal-value');
    const probAttackValue = document.getElementById('prob-attack-value');
    const loading = document.getElementById('loading');
    const initialMessage = document.getElementById('initial-message');
    const resultIcon = document.getElementById('result-icon');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Hide initial message and previous results
        initialMessage.classList.add('hidden');
        resultCard.classList.add('hidden');
        probSection.classList.add('hidden');
        loading.classList.remove('hidden');

        // Collect form data
        const formData = new FormData(form);
        const data = {};
        
        formData.forEach((value, key) => {
            // Keep proto, service, and state as strings, convert others to numbers
            if (key === 'proto' || key === 'state' || key === 'service') {
                data[key] = value;
            } else {
                data[key] = parseFloat(value) || 0;
            }
        });

        console.log('Sending data:', data);  // Debug log

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            
            // Hide loading
            loading.classList.add('hidden');

            if (result.error) {
                alert('Error: ' + result.error);
                initialMessage.classList.remove('hidden');
                return;
            }

            // Show result card
            resultCard.classList.remove('hidden', 'normal', 'attack');
            
            if (result.prediction === 1) {
                resultCard.classList.add('attack');
                resultIcon.innerHTML = '⚠️';
                resultLabel.textContent = 'ATTACK DETECTED!';
            } else {
                resultCard.classList.add('normal');
                resultIcon.innerHTML = '✅';
                resultLabel.textContent = 'Normal Traffic';
            }
            
            resultConfidence.textContent = `Confidence: ${result.confidence.toFixed(2)}%`;

            // Show probability bars
            probSection.classList.remove('hidden');
            
            // Reset bars first
            probNormal.style.width = '0%';
            probAttack.style.width = '0%';
            
            // Animate probability bars
            setTimeout(() => {
                probNormal.style.width = result.prob_normal + '%';
                probAttack.style.width = result.prob_attack + '%';
                probNormalValue.textContent = result.prob_normal.toFixed(1) + '%';
                probAttackValue.textContent = result.prob_attack.toFixed(1) + '%';
            }, 100);

        } catch (error) {
            loading.classList.add('hidden');
            initialMessage.classList.remove('hidden');
            alert('Error connecting to server: ' + error.message);
        }
    });

    // Add input validation feedback
    const inputs = form.querySelectorAll('input[type="number"]');
    inputs.forEach(input => {
        input.addEventListener('input', function() {
            if (this.value < 0) {
                this.value = 0;
            }
        });
    });

    // Load Sample Data functionality
    const loadNormalBtn = document.getElementById('load-normal');
    const loadAttackBtn = document.getElementById('load-attack');
    const sampleInfo = document.getElementById('sample-info');

    async function loadSample(type) {
        try {
            sampleInfo.textContent = 'Loading sample...';
            const response = await fetch(`/sample/${type}`);
            const data = await response.json();
            
            if (data.error) {
                sampleInfo.textContent = 'Error: ' + data.error;
                return;
            }
            
            // Fill in the form with sample data
            for (const [key, value] of Object.entries(data)) {
                const input = document.getElementById(key);
                if (input) {
                    if (input.tagName === 'SELECT') {
                        input.value = value;
                    } else {
                        input.value = value;
                    }
                }
            }
            
            sampleInfo.textContent = `Loaded ${data.actual_label} sample - click "Detect Intrusion" to test`;
            sampleInfo.style.color = type === 'normal' ? '#48bb78' : '#f56565';
            
        } catch (error) {
            sampleInfo.textContent = 'Error loading sample: ' + error.message;
        }
    }

    loadNormalBtn.addEventListener('click', () => loadSample('normal'));
    loadAttackBtn.addEventListener('click', () => loadSample('attack'));
});