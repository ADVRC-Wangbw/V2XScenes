(function () {
  const config = window.V2XSCENES_REQUEST_CONFIG || {};
  const form = document.getElementById("request-form");
  const frame = document.getElementById("submission-frame");
  const button = document.getElementById("submit-button");
  const status = document.getElementById("form-status");
  const submittedAtClient = document.getElementById("submittedAtClient");

  let hasSubmitted = false;

  function setStatus(message, type) {
    status.textContent = message;
    status.className = type ? `status ${type}` : "status";
  }

  if (!config.endpoint || config.endpoint.includes("PASTE_GOOGLE_APPS_SCRIPT")) {
    button.disabled = true;
    setStatus("The request form is not configured yet. Please contact the maintainer.", "error");
    return;
  }

  form.action = config.endpoint;

  form.addEventListener("submit", function (event) {
    setStatus("", "");

    if (!form.checkValidity()) {
      event.preventDefault();
      form.reportValidity();
      return;
    }

    if (form.elements.website.value) {
      event.preventDefault();
      setStatus("Submission failed. Please refresh the page and try again.", "error");
      return;
    }

    submittedAtClient.value = new Date().toISOString();
    hasSubmitted = true;
    button.disabled = true;
    button.textContent = "Submitting...";
    setStatus("Submitting your request. Please wait.");
  });

  frame.addEventListener("load", function () {
    if (!hasSubmitted) return;

    hasSubmitted = false;
    form.reset();
    button.disabled = false;
    button.textContent = "Submit Request";
    setStatus("Your request has been submitted. If approved, the download code link will be sent to your email.", "success");
  });
})();
