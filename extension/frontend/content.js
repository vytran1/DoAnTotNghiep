// chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
//   if (request.message === "get_page_text") {
//     let article = document.querySelector("article");
//     let textContent = "";

//     if (article) {
//       textContent = Array.from(article.querySelectorAll("p"))
//         .map((p) => p.innerText)
//         .join("\n");
//     } else {
//       textContent = document.body.innerText;
//     }

//     sendResponse({ text: textContent });
//   }
//   return true;
// });

// chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
//   if (request.message === "get_page_text") {
//     // Lấy tất cả từ toàn bộ trang
//     let allContent = Array.from(
//       document.querySelectorAll(
//         "header, h1, h2, h3, h4, h5, h6, title, article p, p"
//       )
//     )
//       .map((el) => el.innerText.trim())
//       .filter((text) => text.length > 0)
//       .join("\n\n");

//     sendResponse({ text: allContent });
//   }
//   return true;
// });

// chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
//   if (request.message === "get_page_text") {
//     let allContent = "";

//     // Tìm article hoặc main content
//     let mainContent =
//       document.querySelector("article") ||
//       document.querySelector("main") ||
//       document.querySelector('[role="main"]') ||
//       document.querySelector(".content");

//     if (mainContent) {
//       // Lấy TẤT CẢ elements có text
//       allContent = Array.from(
//         mainContent.querySelectorAll(
//           "h1, h2, h3, h4, h5, h6, p, li, span, div, blockquote, pre"
//         )
//       )
//         .map((el) => el.innerText.trim())
//         .filter((text) => text.length > 10) // Lọc text quá ngắn
//         .filter((text, index, arr) => arr.indexOf(text) === index) // Loại duplicate
//         .join("\n\n");
//     } else {
//       // Fallback
//       allContent = document.body.innerText;
//     }

//     sendResponse({
//       text: allContent,
//       url: window.location.href,
//     });
//   }
//   return true;
// });

// chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
//   if (request.message === "get_page_text") {
//     // Lấy tất cả từ toàn bộ trang
//     let allContent = Array.from(
//       document.querySelectorAll(
//         "header, h1, h2, h3, h4, h5, h6, title, article p, p"
//       )
//     )
//       .map((el) => {
//         // Kiểm tra innerText hoặc textContent
//         return (el.innerText || el.textContent || "").trim();
//       })
//       .filter((text) => text.length > 0)
//       .join("\n\n");

//     sendResponse({ text: allContent });
//   }
//   return true;
// });

// chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
//   if (request.message === "get_page_text") {
//     // 1. Lấy tiêu đề chính
//     const title = document.querySelector("h1")?.innerText?.trim() || "";

//     // 2. Tìm article container
//     const article =
//       document.querySelector("article") ||
//       document.querySelector('[role="main"]') ||
//       document.querySelector("main") ||
//       document.body;

//     // 3. Lấy các đoạn văn, loại bỏ navigation/ads/footer
//     const paragraphs = Array.from(article.querySelectorAll("p"))
//       .filter((p) => {
//         // Loại bỏ các thẻ p trong nav, footer, aside, ads
//         const parent = p.closest(
//           "nav, footer, aside, .ad, .advertisement, .sidebar, .related-posts"
//         );
//         return !parent;
//       })
//       .map((p) => p.innerText?.trim() || "")
//       .filter((text) => {
//         // Chỉ lấy đoạn văn có độ dài hợp lý (loại bỏ "Read more", "Share", etc.)
//         return (
//           text.length > 50 &&
//           !text.match(/^(Share|Read more|Subscribe|Follow)/i)
//         );
//       })
//       .join("\n\n");

//     const fullContent = title ? `${title}\n\n${paragraphs}` : paragraphs;

//     console.log(`📊 Extracted ${paragraphs.split("\n\n").length} paragraphs`);
//     sendResponse({ text: fullContent });
//   }
//   return true;
// });

// chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
//   if (request.message === "get_page_text") {
//     function extractArticleContent() {
//       let content = [];

//       // 1. LẤY TIÊU ĐỀ CHÍNH
//       const mainTitle = document.querySelector(
//         'h1, .c-title, [class*="title"]'
//       );
//       if (mainTitle) {
//         content.push(mainTitle.innerText.trim());
//       }

//       // 2. LẤY CLAIM/STATEMENT (đặc trưng của fact-check sites)
//       const claim = document.querySelector(
//         '.m-statement__quote, [class*="claim"], [class*="statement"]'
//       );
//       if (claim) {
//         content.push(claim.innerText.trim());
//       }

//       // 3. TÌM ARTICLE CONTAINER
//       // Thứ tự ưu tiên: article tag > main tag > id/class có chứa "article"
//       let articleContainer =
//         document.querySelector("article.m-textblock") ||
//         document.querySelector("article") ||
//         document.querySelector("main") ||
//         document.querySelector('[class*="article-content"]') ||
//         document.querySelector('[id*="article"]') ||
//         document.querySelector('[class*="content"][class*="main"]');

//       if (!articleContainer) {
//         // Fallback: tìm div có nhiều paragraph nhất
//         const allDivs = document.querySelectorAll("div");
//         let maxParagraphs = 0;

//         allDivs.forEach((div) => {
//           const paragraphs = div.querySelectorAll("p");
//           if (paragraphs.length > maxParagraphs) {
//             maxParagraphs = paragraphs.length;
//             articleContainer = div;
//           }
//         });
//       }

//       if (articleContainer) {
//         // 4. LẤY TẤT CẢ PARAGRAPHS TRONG ARTICLE
//         const paragraphs = articleContainer.querySelectorAll("p");

//         paragraphs.forEach((p) => {
//           // Loại bỏ các paragraph trong nav, footer, sidebar, ads
//           const isInExcludedArea = p.closest(
//             'nav, footer, aside, header, [class*="ad"], [class*="sidebar"], [class*="related"], [class*="menu"], [class*="social"]'
//           );

//           if (!isInExcludedArea) {
//             const text = p.innerText.trim();

//             // Chỉ lấy đoạn văn có độ dài hợp lý (> 50 ký tự)
//             // Loại bỏ các text như "Read more", "Share", "Subscribe"
//             if (
//               text.length > 50 &&
//               !text.match(
//                 /^(Share|Read more|Subscribe|Follow|Sign up|Click here|Advertisement)/i
//               )
//             ) {
//               content.push(text);
//             }
//           }
//         });
//       }

//       // 5. LẤY "IF YOUR TIME IS SHORT" SUMMARY (nếu có)
//       const summary = document.querySelector(
//         '.short-on-time, [class*="summary"]'
//       );
//       if (summary) {
//         const summaryText = summary.innerText.trim();
//         if (summaryText && !content.includes(summaryText)) {
//           content.unshift(summaryText); // Thêm vào đầu
//         }
//       }

//       // 6. KẾT HỢP THÀNH TEXT HOÀN CHỈNH
//       const finalText = content.join("\n\n");

//       console.log(`✅ Extracted ${content.length} sections`);
//       console.log(`📄 Total length: ${finalText.length} characters`);

//       return finalText;
//     }

//     // GỌI HÀM VÀ TRẢ VỀ KẾT QUẢ
//     const articleContent = extractArticleContent();
//     sendResponse({ text: articleContent });
//   }
//   return true;
// });

// chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
//   if (request.message === "get_page_text") {
//     console.log("🔍 Starting Readability extraction...");

//     try {
//       // Clone document
//       const documentClone = document.cloneNode(true);

//       // Khởi tạo Readability
//       const reader = new Readability(documentClone);

//       // Parse article
//       const article = reader.parse();

//       if (article) {
//         console.log("✅ Readability SUCCESS!");
//         console.log(`📰 Title: ${article.title}`);
//         console.log(`📝 Content length: ${article.textContent.length} chars`);

//         // Kết hợp title + content
//         const fullContent = `${article.title}\n\n${article.textContent}`;
//         console.log("Full Context", fullContent);

//         sendResponse({ text: fullContent });
//       } else {
//         console.warn("⚠️ Readability returned null, using fallback");
//         sendResponse({ text: document.body.innerText });
//       }
//     } catch (error) {
//       console.error("❌ Readability ERROR:", error);
//       sendResponse({ text: document.body.innerText });
//     }
//   }
//   return true;
// });

// chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
//   if (request.message === "get_page_text") {
//     try {
//       // 1. Clone và dọn dẹp sơ bộ trước khi Parse
//       const documentClone = document.cloneNode(true);

//       // 2. Dùng Readability để bóc tách "lõi" nội dung
//       const reader = new Readability(documentClone);
//       const article = reader.parse();

//       if (article && article.textContent) {
//         // 3. Xử lý thành một đoạn duy nhất
//         const cleanedText = article.textContent
//           .replace(/\s+/g, " ") // Thay thế TẤT CẢ các loại khoảng trắng (n, r, t, space) thành 1 dấu cách duy nhất
//           .trim(); // Cắt bỏ khoảng trắng ở 2 đầu

//         // 4. Kết hợp Title và Content thành 1 chuỗi liền mạch
//         const finalResult = `TITLE: ${article.title.trim()} | CONTENT: ${cleanedText}`;

//         console.log("✅ Extraction Complete (Single Paragraph)");
//         sendResponse({ text: finalResult });
//       } else {
//         // Fallback đơn giản nếu Readability thất bại
//         const fallback = document.body.innerText.replace(/\s+/g, " ").trim();
//         sendResponse({ text: fallback });
//       }
//     } catch (error) {
//       console.error("❌ Error:", error);
//       sendResponse({ text: "" });
//     }
//   }
//   return true;
// });

// chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
//   if (request.message === "get_page_text") {
//     // 1. Lấy Title
//     // Ưu tiên lấy thẻ h1, nếu không có thì lấy document.title
//     const titleElement = document.querySelector("h1");
//     const title = titleElement ? titleElement.innerText : document.title;

//     // 2. Lấy Content
//     let article = document.querySelector("article");
//     let textContent = "";

//     if (article) {
//       // Reuters lưu nội dung chính trong các thẻ div có data-testid="paragraph-..."
//       // và phần mô tả (sapo) trong div có data-testid="Dek"
//       const selectors = [
//         'div[data-testid="Dek"]', // Phần mô tả đầu bài
//         'div[data-testid^="paragraph-"]', // Các đoạn văn nội dung chính
//       ];

//       let nodes = article.querySelectorAll(selectors.join(", "));

//       // Nếu không tìm thấy theo cấu trúc đặc thù (trang khác Reuters), fallback về thẻ p
//       if (nodes.length === 0) {
//         nodes = article.querySelectorAll("p");
//       }

//       textContent = Array.from(nodes)
//         .map((node) => node.innerText.trim()) // Lấy text và xóa khoảng trắng thừa
//         .filter((text) => text.length > 0) // Loại bỏ các đoạn rỗng
//         .join("\n\n"); // Nối các đoạn bằng 2 dấu xuống dòng cho dễ đọc
//     } else {
//       // Fallback nếu không tìm thấy thẻ article
//       textContent = document.body.innerText;
//     }

//     // Trả về cả title và text
//     sendResponse({
//       title: title,
//       text: textContent,
//     });
//   }
//   return true;
// });

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.message !== "get_page_text") return;

  try {
    const cloned = document.cloneNode(true);
    const reader = new Readability(cloned);
    const article = reader.parse();

    if (!article || !article.textContent) {
      return sendResponse({
        success: false,
        text: document.body.innerText.trim(),
      });
    }

    const cleaned = article.textContent.replace(/\s+/g, " ").trim();
    const title = article.title?.trim() || document.title;
    const content = article.textContent.replace(/\s+/g, " ").trim();
    const combinedText = `${title}\n\n${content}`;
    console.log("Cleaned Text", cleaned);
    console.log("Article Title", article.title?.trim() || document.title);
    console.log("Combined Text", combinedText);

    sendResponse({
      success: true,
      title: article.title?.trim() || document.title,
      text: combinedText,
    });
  } catch (e) {
    console.error("Extractor error", e);
    sendResponse({
      success: false,
      text: document.body.innerText.trim(),
    });
  }

  return true;
});
