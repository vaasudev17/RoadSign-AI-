const dropArea = document.getElementById("drop-area");

const fileInput = document.getElementById("file-input");

const preview = document.getElementById("preview");

const uploadContent =
    document.getElementById("upload-content");

const scanButton =
    document.getElementById("scan-button");

const resultSection =
    document.getElementById("result-section");

const predictionElement =
    document.getElementById("prediction");

const confidenceElement =
    document.getElementById("confidence");

const confidenceCircle =
    document.getElementById("confidence-circle");


let selectedFile = null;


// --------------------------------------------------
// CLICK TO SELECT IMAGE
// --------------------------------------------------

dropArea.addEventListener("click", () => {

    fileInput.click();

});


// --------------------------------------------------
// FILE SELECTED
// --------------------------------------------------

fileInput.addEventListener("change", (event) => {

    const file = event.target.files[0];

    if (file) {

        handleFile(file);

    }

});


// --------------------------------------------------
// DRAG & DROP
// --------------------------------------------------

dropArea.addEventListener("dragover", (event) => {

    event.preventDefault();

    dropArea.classList.add("dragging");

});


dropArea.addEventListener("dragleave", () => {

    dropArea.classList.remove("dragging");

});


dropArea.addEventListener("drop", (event) => {

    event.preventDefault();

    dropArea.classList.remove("dragging");

    const file = event.dataTransfer.files[0];

    if (file) {

        handleFile(file);

    }

});


// --------------------------------------------------
// HANDLE IMAGE
// --------------------------------------------------

function handleFile(file) {

    if (!file.type.startsWith("image/")) {

        alert("Please select an image file.");

        return;

    }


    selectedFile = file;


    const reader = new FileReader();


    reader.onload = function(event) {

        preview.src = event.target.result;

        preview.style.display = "block";

        uploadContent.style.display = "none";

        scanButton.disabled = false;

    };


    reader.readAsDataURL(file);

}


// --------------------------------------------------
// SCAN
// --------------------------------------------------

scanButton.addEventListener("click", async () => {

    if (!selectedFile) {

        return;

    }


    scanButton.disabled = true;

    scanButton.innerHTML = "ANALYZING...";


    const formData = new FormData();

    formData.append("image", selectedFile);


    try {

        const response = await fetch(
            "/predict",
            {
                method: "POST",
                body: formData
            }
        );


        const data = await response.json();


        if (!response.ok || data.error) {

            throw new Error(
                data.error || "Prediction failed"
            );

        }


        // Best prediction only

        predictionElement.textContent =
            data.prediction;


        const confidence =
            Number(data.confidence);


        confidenceElement.textContent =
            confidence.toFixed(2) + "%";


        // Update confidence circle

        const degrees =
            confidence * 3.6;


        confidenceCircle.style.background = `
            radial-gradient(
                circle at center,
                #0b1916 62%,
                transparent 63%
            ),
            conic-gradient(
                #b8ff62 ${degrees}deg,
                #20372f ${degrees}deg
            )
        `;


        resultSection.classList.remove("hidden");


        resultSection.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });


    }

    catch (error) {

        alert(
            "Prediction error: " +
            error.message
        );

    }


    scanButton.disabled = false;

    scanButton.innerHTML =
        'SCAN SIGN <span>→</span>';

});