const imageUpload = document.getElementById('imageUpload');
const predictButton = document.getElementById('predictButton');
const uploadedImage = document.getElementById('uploadedImage');
const placeholderText = document.getElementById('placeholderText');
const predictionResult = document.getElementById('predictionResult');

let session; // ONNX Runtime session

// Load the ONNX model
async function loadModel() {
    predictionResult.textContent = 'Loading model...';
    try {
        session = await ort.InferenceSession.create('mobilenet_v2_ai_real_embedded.onnx');
        predictionResult.textContent = 'Model loaded successfully. Upload an image to start.';
        console.log('ONNX model loaded successfully');
    } catch (e) {
        console.error('Failed to load ONNX model:', e);
        predictionResult.textContent = `Error loading model: ${e.message}`;
        predictionResult.classList.add('error');
    }
}

// Preprocess the image to fit the model's input requirements
// This should mirror the transforms applied in Python (Resize, Sharpen, ToTensor, Normalize)
async function preprocessImage(imageElement) {
    const width = 224;
    const height = 224;
    const image = new Image();
    image.src = imageElement.src;

    await new Promise(resolve => {
        image.onload = resolve;
    });

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');

    // Resize and draw image
    ctx.drawImage(image, 0, 0, width, height);

    // Sharpening (optional and complex in JS, might skip for simplicity or use a library)
    // For a real-world scenario, you might need a custom WebGL shader or a more advanced library for sharpening.
    // Here, we'll focus on Resize, ToTensor, Normalize.

    const imageData = ctx.getImageData(0, 0, width, height).data;

    // Convert to Tensor and Normalize
    const red = [], green = [], blue = [];
    for (let i = 0; i < imageData.length; i += 4) {
        // Normalize to [0, 1] then subtract mean and divide by std
        red.push((imageData[i] / 255 - 0.485) / 0.229);
        green.push((imageData[i + 1] / 255 - 0.456) / 0.224);
        blue.push((imageData[i + 2] / 255 - 0.406) / 0.225);
    }

    // Concatenate R, G, B channels
    const inputData = [...red, ...green, ...blue];
    const inputTensor = new ort.Tensor('float32', inputData, [1, 3, height, width]);

    return inputTensor;
}

// Run inference
async function runInference(inputTensor) {
    if (!session) {
        console.error('ONNX session not loaded.');
        predictionResult.textContent = 'Error: Model not loaded.';
        predictionResult.classList.add('error');
        return;
    }

    try {
        const feeds = { input: inputTensor }; // 'input' matches the input_names in export
        const results = await session.run(feeds);
        const output = results[session.outputNames[0]]; // Get the output tensor

        // Assuming a 2-class output, get probabilities or logits
        // For simplicity, let's assume it's logits and we take argmax
        const outputArray = output.data; // Float32Array

        // Softmax to get probabilities (if output is logits)
        const expOutput = outputArray.map(Math.exp);
        const sumExpOutput = expOutput.reduce((a, b) => a + b, 0);
        const probabilities = expOutput.map(val => val / sumExpOutput);

        const classLabels = ["Real Image", "AI Generated Image"];
        const predictedClassIndex = probabilities[0] > probabilities[1] ? 0 : 1; // Assuming binary classification
        const predictedLabel = classLabels[predictedClassIndex];
        const confidence = Math.max(probabilities[0], probabilities[1]) * 100;

        predictionResult.textContent = `Prediction: ${predictedLabel} (${confidence.toFixed(2)}%)`;
        predictionResult.className = 'result-area success';
        console.log('Prediction:', predictedLabel, 'Confidence:', confidence.toFixed(2) + '%');

    } catch (e) {
        console.error('Failed to run inference:', e);
        predictionResult.textContent = `Error during prediction: ${e.message}`;
        predictionResult.classList.add('error');
    }
}

// Event Listeners
imageUpload.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadedImage.src = e.target.result;
            uploadedImage.style.display = 'block';
            placeholderText.style.display = 'none';
            predictButton.disabled = false;
            predictionResult.textContent = ''; // Clear previous results
            predictionResult.classList.remove('success', 'error');
        };
        reader.readAsDataURL(file);
    } else {
        uploadedImage.style.display = 'none';
        placeholderText.style.display = 'block';
        predictButton.disabled = true;
        predictionResult.textContent = 'Please upload an image.';
    }
});

predictButton.addEventListener('click', async () => {
    if (uploadedImage.src && uploadedImage.src !== '#') {
        predictButton.disabled = true; // Disable button during prediction
        predictionResult.textContent = 'Predicting...';
        predictionResult.classList.remove('success', 'error');
        try {
            const inputTensor = await preprocessImage(uploadedImage);
            await runInference(inputTensor);
        } finally {
            predictButton.disabled = false; // Re-enable button after prediction
        }
    } else {
        predictionResult.textContent = 'Please upload an image first.';
        predictionResult.classList.add('error');
    }
});

// Initialize model loading when the page loads
window.addEventListener('load', loadModel);
