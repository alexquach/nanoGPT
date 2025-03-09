// Main visualization functionality

// DOM elements
const neuronSelect = document.getElementById('neuron-select');
const topNSlider = document.getElementById('top-n-slider');
const topNValue = document.getElementById('top-n-value');
const selectedNeuronSpan = document.getElementById('selected-neuron');
const modelNameSpan = document.getElementById('model-name');
const samplesContainer = document.getElementById('samples-container');
const modelButtons = document.querySelectorAll('.model-button');

// Current state
let selectedNeuron = neuronSelect ? neuronSelect.value : '';
let topN = parseInt(topNSlider ? topNSlider.value : 10);
let contextWindow = 15;
let significanceThreshold = 0.1;
let neuronStatus = {};
let currentModel = activeModel || 'base'; // This comes from the template

// Initialize the visualization
function init() {
    // Set up model switching buttons
    modelButtons.forEach(button => {
        button.addEventListener('click', (e) => {
            const modelKey = e.target.dataset.model;
            if (modelKey !== currentModel) {
                switchModel(modelKey);
            }
        });
    });
    
    // Set up event listeners
    if (neuronSelect) {
        neuronSelect.addEventListener('change', (e) => {
            selectedNeuron = e.target.value;
            updateVisualization();
        });
    }
    
    if (topNSlider) {
        topNSlider.addEventListener('input', (e) => {
            topN = parseInt(e.target.value);
            topNValue.textContent = topN;
            updateVisualization();
        });
    }
    
    // Add event listener for context slider
    const contextSlider = document.getElementById('context-slider');
    const contextValue = document.getElementById('context-value');
    
    if (contextSlider) {
        contextSlider.addEventListener('input', (e) => {
            contextWindow = parseInt(e.target.value);
            contextValue.textContent = contextWindow;
            updateVisualization();
        });
    }
    
    // Add event listener for threshold slider
    const thresholdSlider = document.getElementById('threshold-slider');
    const thresholdValue = document.getElementById('threshold-value');
    
    if (thresholdSlider) {
        thresholdSlider.addEventListener('input', (e) => {
            significanceThreshold = parseInt(e.target.value) / 100;
            thresholdValue.textContent = significanceThreshold.toFixed(2);
            updateVisualization();
        });
    }
    
    // Set initial values
    if (neuronSelect) {
        selectedNeuron = neuronSelect.value;
    }
    if (selectedNeuronSpan) {
        selectedNeuronSpan.textContent = selectedNeuron;
    }
    
    // Add event listener for threshold changes to update neuron status
    if (thresholdSlider) {
        thresholdSlider.addEventListener('change', () => {
            // 'change' fires when slider interaction ends, not during sliding
            loadNeuronStatus();
        });
    }
    
    // Load neuron status first, then load visualization
    loadNeuronStatus().then(() => {
        updateVisualization();
    });
}

// Switch to a different model
async function switchModel(modelKey) {
    try {
        // Show loading state
        if (samplesContainer) {
            samplesContainer.innerHTML = '<p>Loading model data...</p>';
        }
        
        // Call API to switch model
        const response = await fetch(`/api/switch_model/${modelKey}`);
        const data = await response.json();
        
        if (data.success) {
            // Update UI elements
            currentModel = modelKey;
            
            // Update model buttons
            modelButtons.forEach(button => {
                if (button.dataset.model === modelKey) {
                    button.classList.add('active');
                } else {
                    button.classList.remove('active');
                }
            });
            
            // Update model name
            if (modelNameSpan) {
                modelNameSpan.textContent = data.model;
            }
            
            // Rebuild neuron dropdown
            if (neuronSelect) {
                // Save scroll position
                const scrollPos = window.scrollY;
                
                // Clear existing options
                neuronSelect.innerHTML = '';
                
                // Add new options for this model
                data.neurons.forEach(neuron => {
                    const option = document.createElement('option');
                    option.value = neuron;
                    option.textContent = neuron;
                    neuronSelect.appendChild(option);
                });
                
                // Set the first neuron as selected
                if (neuronSelect.options.length > 0) {
                    selectedNeuron = neuronSelect.options[0].value;
                    if (selectedNeuronSpan) {
                        selectedNeuronSpan.textContent = selectedNeuron;
                    }
                }
                
                // Restore scroll position
                window.scrollTo(0, scrollPos);
            }
            
            // Load neuron status and visualize
            await loadNeuronStatus();
            updateVisualization();
        } else {
            // Show error
            if (samplesContainer) {
                samplesContainer.innerHTML = `<p>Error: ${data.error || 'Failed to switch model'}</p>`;
            }
        }
    } catch (error) {
        console.error('Error switching model:', error);
        if (samplesContainer) {
            samplesContainer.innerHTML = '<p>Error switching model. Please try again.</p>';
        }
    }
}

// Update visualization based on selected neuron
async function updateVisualization() {
    if (selectedNeuronSpan) {
        selectedNeuronSpan.textContent = selectedNeuron;
    }
    
    // Show loading state
    if (samplesContainer) {
        samplesContainer.innerHTML = '<p>Loading...</p>';
    }
    
    try {
        // Fetch data for the selected neuron
        const response = await fetch(`/api/neuron/${selectedNeuron}?top_n=${topN}`);
        const data = await response.json();
        
        // Render the samples
        renderSamples(data);
    } catch (error) {
        console.log(error);
        console.error('Error fetching neuron data:', error);
        if (samplesContainer) {
            samplesContainer.innerHTML = '<p>Error loading data. Please try again.</p>';
        }
    }
}

// Render the samples with highlighted tokens
function renderSamples(samples) {
    if (!samples || samples.length === 0) {
        samplesContainer.innerHTML = '<p>No data available for this neuron.</p>';
        return;
    }
    
    // Clear previous samples
    samplesContainer.innerHTML = '';
    
    // Find the global max activation for better color scaling
    const maxActivation = Math.max(...samples.map(s => s.max_activation));
    
    // Create elements for each sample
    samples.forEach((sample, index) => {
        const sampleCard = document.createElement('div');
        sampleCard.className = 'sample-card';
        
        // Create sample header with metadata
        const header = document.createElement('div');
        header.className = 'sample-header';
        header.innerHTML = `
            <div>
                <strong>Sample ${sample.sample_idx}</strong>
            </div>
            <div>
                <span>Max Activation: ${sample.max_activation.toFixed(4)}</span>
            </div>
        `;
        
        // Group tokens into relevant chunks (paragraph-like units)
        const relevantChunks = extractRelevantChunks(sample.tokens);
        
        // Create the tokens container
        const tokensContainer = document.createElement('div');
        tokensContainer.className = 'sample-tokens';
        
        if (relevantChunks.length === 0) {
            tokensContainer.innerHTML = '<p><em>No significant activations found in this sample.</em></p>';
        } else {
            // Add each chunk with a separator
            relevantChunks.forEach((chunk, chunkIndex) => {
                // Add a separator between chunks
                if (chunkIndex > 0) {
                    const separator = document.createElement('div');
                    separator.className = 'chunk-separator';
                    separator.innerHTML = '⋯⋯⋯⋯⋯';
                    tokensContainer.appendChild(separator);
                }
                
                // Create a container for this chunk
                const chunkContainer = document.createElement('div');
                chunkContainer.className = 'token-chunk';
                
                // Add each token in the chunk
                chunk.forEach(token => {
                    const tokenSpan = document.createElement('span');
                    tokenSpan.className = 'token';
                    tokenSpan.textContent = token.token;
                    
                    // Normalize activation relative to max and assign highlight level
                    const activation = token.activation || 0;
                    const normalizedActivation = activation / maxActivation;
                    
                    if (activation > 0) {
                        // Determine highlight level (0-5)
                        const highlightLevel = Math.ceil(normalizedActivation * 5);
                        tokenSpan.className += ` highlight-level-${highlightLevel}`;
                        
                        // Add tooltip with activation value - include more detail in the tooltip
                        tokenSpan.title = `Activation: ${activation.toFixed(6)}`;
                    }
                    
                    chunkContainer.appendChild(tokenSpan);
                });
                
                tokensContainer.appendChild(chunkContainer);
            });
        }
        
        // Assemble the sample card
        sampleCard.appendChild(header);
        sampleCard.appendChild(tokensContainer);
        
        // Add the sample card to the container
        samplesContainer.appendChild(sampleCard);
    });
}

// Extract relevant chunks of tokens containing activations with context before and after
function extractRelevantChunks(tokens) {
    const chunks = [];
    const significantThreshold = significanceThreshold;
    
    // First, identify all significant tokens and mark their indices
    const significantIndices = [];
    tokens.forEach((token, index) => {
        const activation = token.activation || 0;
        if (activation > significantThreshold) {
            significantIndices.push(index);
        }
    });
    
    // If no significant activations, return empty array
    if (significantIndices.length === 0) {
        return [];
    }
    
    // Build chunks around significant tokens with context before and after
    let currentChunkStart = Math.max(0, significantIndices[0] - contextWindow);
    let currentChunkEnd = significantIndices[0] + contextWindow;
    
    // Process all significant tokens to create chunks
    for (let i = 1; i < significantIndices.length; i++) {
        const tokenIndex = significantIndices[i];
        const chunkStartCandidate = Math.max(0, tokenIndex - contextWindow);
        const chunkEndCandidate = Math.min(tokens.length - 1, tokenIndex + contextWindow);
        
        // If this significant token is close to the previous chunk, extend the current chunk
        if (chunkStartCandidate <= currentChunkEnd + 1) {
            currentChunkEnd = chunkEndCandidate;
        } else {
            // Otherwise, finish the current chunk and start a new one
            chunks.push(tokens.slice(currentChunkStart, currentChunkEnd + 1));
            currentChunkStart = chunkStartCandidate;
            currentChunkEnd = chunkEndCandidate;
        }
    }
    
    // Add the final chunk
    chunks.push(tokens.slice(currentChunkStart, currentChunkEnd + 1));
    
    return chunks;
}

// Load neuron status data
async function loadNeuronStatus() {
    try {
        const response = await fetch(`/api/neuron_status?threshold=${significanceThreshold}`);
        neuronStatus = await response.json();
        updateNeuronDropdown();
    } catch (error) {
        console.error('Error loading neuron status:', error);
    }
}

// Update the neuron dropdown with activation status
function updateNeuronDropdown() {
    const select = document.getElementById('neuron-select');
    if (!select) return;
    
    // Save the currently selected value
    const currentValue = select.value;
    
    // Clear existing options
    select.innerHTML = '';
    
    // Add options with status indicators
    for (const neuronId in neuronStatus) {
        if (neuronStatus.hasOwnProperty(neuronId)) {
            const option = document.createElement('option');
            option.value = neuronId;
            
            const status = neuronStatus[neuronId];
            if (status.has_activations) {
                // Add a visual indicator for active neurons
                option.textContent = `${neuronId} ✓ (${status.max_activation.toFixed(2)})`;
                option.className = 'active-neuron';
            } else {
                option.textContent = `${neuronId} ○`;
                option.className = 'inactive-neuron';
            }
            
            select.appendChild(option);
        }
    }
    
    // Restore the previously selected value if possible
    if (currentValue && Array.from(select.options).some(opt => opt.value === currentValue)) {
        select.value = currentValue;
    } else if (select.options.length > 0) {
        // Select the first option if the previous value is no longer available
        select.selectedIndex = 0;
        selectedNeuron = select.value;
    }
}

// Initialize the visualization when the page loads
document.addEventListener('DOMContentLoaded', init); 