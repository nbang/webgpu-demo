import { AutoModelForVision2Seq, AutoProcessor, RawImage, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers';

// Configure environment
env.allowLocalModels = false;
env.useBrowserCache = true;

const MODEL_ID = 'HuggingFaceTB/SmolVLM-256M-Instruct';

let model = null;
let processor = null;

const statusEl = document.getElementById('status');
const fileInput = document.getElementById('file-input');
const previewContainer = document.getElementById('preview-container');
const resultTextarea = document.getElementById('result');

function setStatus(message, type = 'info') {
    statusEl.innerHTML = message;
    statusEl.className = `alert alert-${type} text-center fw-semibold mx-auto`;
    statusEl.style.maxWidth = '800px';
}

async function init() {
    try {
        setStatus('<i class="bi bi-download me-2"></i>Loading SmolVLM model...', 'info');

        const device = 'webgpu';
        processor = await AutoProcessor.from_pretrained(MODEL_ID);

        model = await AutoModelForVision2Seq.from_pretrained(MODEL_ID, {
            device: device,
            dtype: {
                embed_tokens: 'fp32',
                vision_encoder: 'fp32',
                encoder_model: 'q4',
                decoder_model_merged: 'q4',
            },
        });

        setStatus('<i class="bi bi-check-circle-fill me-2"></i>Ready! Select an image.', 'success');
        document.getElementById('file-input').disabled = false;

    } catch (e) {
        console.error(e);
        setStatus(`<i class="bi bi-exclamation-triangle-fill me-2"></i>Error: ${e.message}`, 'danger');
    }
}

async function runOCR(imageUrl) {
    if (!model || !processor) return;

    try {
        setStatus('<i class="bi bi-cpu me-2 animate-spin"></i>Processing...', 'warning');
        resultTextarea.value = '';

        const image = await RawImage.fromURL(imageUrl);

        // Prepare conversation for SmolVLM
        const messages = [
            {
                role: "user",
                content: [
                    { type: "image" },
                    { type: "text", text: "Extract all text from this image." }
                ]
            }
        ];

        // Format inputs
        const text_inputs = processor.apply_chat_template(messages, { render_bos_token: false });
        const inputs = await processor(text_inputs, [image]);

        // Generate response
        const outputs = await model.generate({
            ...inputs,
            max_new_tokens: 1024,
            do_sample: false,
            repetition_penalty: 1.1,
        });

        // Decode result
        const generatedFullText = processor.decode(outputs[0], { skip_special_tokens: true });
        const promptText = processor.decode(inputs.input_ids[0], { skip_special_tokens: true });

        const cleanText = generatedFullText.replace(promptText, '').replace(/^A:\s*/, '').trim();
        resultTextarea.value = cleanText;
        setStatus('<i class="bi bi-check-circle-fill me-2"></i>Done!', 'success');
    } catch (e) {
        console.error(e);
        setStatus(`Error: ${e.message}`, 'danger');
    }
}

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
        const url = event.target.result;
        previewContainer.innerHTML = `<img id="preview" src="${url}" style="max-width:100%; max-height:500px">`;
        runOCR(url);
    };
    reader.readAsDataURL(file);
});



// Start initialization
init();
