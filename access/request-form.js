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
    setStatus("表单尚未配置提交地址，请联系维护者。", "error");
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
      setStatus("提交失败，请刷新页面后重试。", "error");
      return;
    }

    submittedAtClient.value = new Date().toISOString();
    hasSubmitted = true;
    button.disabled = true;
    button.textContent = "提交中...";
    setStatus("正在提交申请，请稍候。");
  });

  frame.addEventListener("load", function () {
    if (!hasSubmitted) return;

    hasSubmitted = false;
    form.reset();
    button.disabled = false;
    button.textContent = "提交申请";
    setStatus("申请已提交。审核通过后，下载代码链接会发送到你的邮箱。", "success");
  });
})();
