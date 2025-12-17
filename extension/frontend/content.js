chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  if (request.message === "get_page_text") {
    let article = document.querySelector("article");
    let textContent = "";

    if (article) {
      textContent = Array.from(article.querySelectorAll("p"))
        .map((p) => p.innerText)
        .join("\n");
    } else {
      textContent = document.body.innerText;
    }

    sendResponse({ text: textContent });
  }
  return true;
});
