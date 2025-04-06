# Initial preparation
This section aims to mention some problems you may meet when you deal with the data, especially the displayed issues.

When you open the downloaded csv files, some garbled texts will be displayed due to the decoding issues of excel. There are two method to deal with this problem.

- Use Vscode to open it
- Import feature in excel
As you open the excel, there is a **Data** bottom beside the **Start** one. Click it and choose the **From file/CSV** below it. Choose the one you want to open. Next, you need to check whether the utf-8 encoding is applied. Here is the detail below the title **original format of file** (65001: Unicode(UTF-8))

Another thing required to consider

Since the given data has large amount which exceeds the maximum displayed rows, the filter function may not return a correct answer to you. Using programs to manipulate the table is the one I recommend. 

Back to the [Readme file](\README.md)
