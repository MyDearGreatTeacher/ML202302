# 期中考報告格式與內容
- 1.DATA SCIENCE資料科學
- 2.資料科學底層核心技術NUMPY
- 3.資料視覺化(Data Visualization)
- 4.Pandas技術實戰 

# 1.DATA SCIENCE資料科學
# 2.資料科學底層核心技術NUMPY
- NUMPY
- ndarray資料結構與屬性
- ndarray的各項運算
- NUMPY的模組
  - numpy.random
- NUMPY的進階主題
# 3.資料視覺化(Data Visualization)
# matplotlib
- 資料視覺化(Data Visualization) [Data and information visualization](https://en.wikipedia.org/wiki/Data_and_information_visualization)
- 資料視覺化常用工具
- matplotlib簡介與學習資源
- matplotlib畫圖架構
- matplotlib套件
- matplotlib功能展示
- 單一圖表的顯示技術[以折現圖加以說明](./2_1_Matplotlib.md)
  - 線寬 linewidth(lw)
  - 線條樣式 linestyle(ls) [linestyle](https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html) [中文說明](https://blog.csdn.net/Strive_For_Future/article/details/118755312)
  - 顏色配置 color
  - 節點的樣式 marker
  - 標題 | x 軸 | y 軸
  - 途中的文字與數學公式顯示技術 [Text in Matplotlib Plots]()
  - legend() 
- 各種統計圖表的顯示技術 [Statistics plots](https://matplotlib.org/stable/tutorials/text/text_intro.html)
  - plot() 折線圖 == 了解資料趨勢
  - scatter() 散布圖 == 了解資料相關度
  - bar() 柱狀圖
  - barh() 條形圖
  - hist()直方圖 [hist(x)](https://matplotlib.org/stable/plot_types/stats/hist_plot.html)
  - pie()圓餅圖 [pie(x)](https://matplotlib.org/stable/plot_types/stats/pie.html)
  - polar()極線圖 
  - stem()——用於繪製棉棒圖 
  - boxplot()箱型圖 [boxplot(X)](https://matplotlib.org/stable/plot_types/stats/boxplot_plot.html)
  - errorbar() 誤差棒圖  [errorbar(x, y, yerr, xerr)](https://matplotlib.org/stable/plot_types/stats/errorbar_plot.html)
- 多表並陳的技術
- Interactive Visualization(互動式顯示)技術 

# 4.Pandas技術實戰 
- pandas 資料分析
- pandas的資料結構(Data Structures)與基本屬性 ==> series vs DataFrame
- 建立DataFrame的各種技巧
- pandas資料匯入:如何將資料載入成DataFrame
  - 讀寫CSV檔案 
    - 各式csv的讀取技術
      - 去除頭部說明文字
      - 去除底部說明文字
      - 讀取部分欄位
      - 讀取部分資料 
   - 讀寫excel檔案 Reading and writing data in Excel format
   - 讀寫 JSON 檔案
   - 讀取網頁表格資料 
- pandas資料清理(Data cleaning)
- DataFrame的運算1:
