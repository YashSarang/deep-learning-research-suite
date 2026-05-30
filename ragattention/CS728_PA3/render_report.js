const fs = require('fs');
const path = require('path');
const { marked } = require('marked');
const puppeteer = require('puppeteer');

(async () => {
  try {
    const mdPath = path.join(__dirname, 'final_report.md');
    const cssPath = path.join(__dirname, 'report_style.css');
    const outputPath = path.join(__dirname, 'CS728_PA3_Final_Report.pdf');

    if (!fs.existsSync(mdPath)) throw new Error('final_report.md not found');
    const mdContent = fs.readFileSync(mdPath, 'utf8');
    const cssContent = fs.existsSync(cssPath) ? fs.readFileSync(cssPath, 'utf8') : '';

    // Convert MD to HTML
    const htmlContent = marked(mdContent);

    // Full HTML wrapper
    const fullHtml = `
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="UTF-8">
        <style>
          ${cssContent}
        </style>
      </head>
      <body>
        <div class="content">
          ${htmlContent}
        </div>
      </body>
      </html>
    `;

    // Launch Puppeteer
    const browser = await puppeteer.launch({ 
      headless: "new",
      args: ['--no-sandbox', '--disable-setuid-sandbox'] 
    });
    const page = await browser.newPage();

    // Load content
    // We set the base URL to __dirname so relative images work
    await page.setContent(fullHtml, { 
       waitUntil: 'networkidle0',
       baseURL: `file://${__dirname}/` 
    });

    // Generate PDF
    await page.pdf({
      path: outputPath,
      format: 'A4',
      printBackground: true,
      margin: {
        top: '10mm',
        right: '15mm',
        bottom: '10mm',
        left: '15mm'
      }
    });

    await browser.close();
    console.log('PDF generated successfully at: ' + outputPath);
  } catch (error) {
    console.error('Error generating PDF:', error);
    process.exit(1);
  }
})();
