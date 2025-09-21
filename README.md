# 🚀 Aya Karbich | Project Portfolio  

Welcome to my project showcase!  
This portfolio highlights selected academic projects and professional experiences in AI, Data Engineering, Web Development, and IoT.  

---

## 🎓 Academic Projects  




---


\documentclass[11pt]{article}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{xcolor}

\hypersetup{
  colorlinks=true,
  linkcolor=black,
  urlcolor=blue
}

\setlist[itemize]{leftmargin=*,nosep}
\setlist[description]{leftmargin=1.5em,labelindent=0em,itemsep=0.25em}

\begin{document}

\begin{center}
{\LARGE \textbf{Ask-Your-Data: AI-Powered Data Analysis Platform — \textit{GenAI Analyst}}}\par
\vspace{0.5em}
\end{center}

\section*{Description}
Developed a sophisticated \textbf{AI-powered data analysis platform} that transforms how users interact with their data through natural language processing and automated visualization generation. The system leverages Google's Gemini AI models to interpret user questions, automatically select appropriate analysis methods, and generate publication-ready visualizations, making advanced analytics accessible to non-technical users.

\section*{Key Features}
\begin{itemize}
  \item Natural language data queries with \textbf{Gemini 2.5 Pro/Flash} models
  \item \textbf{15+ automated analysis functions} covering statistics, ML, and visualization
  \item Intelligent dashboard generation with \textbf{4--8 smart visualizations}
  \item Real-time data preprocessing and quality assessment
  \item \textbf{LangChain integration} for structured tool calling and prompt management
  \item Drag-and-drop dashboard customization with interactive plot management
\end{itemize}

\section*{Performance Highlights}
\begin{itemize}
  \item \textbf{Auto Dashboard Generation:} Creates comprehensive dashboards in \textbf{\(\sim\)5--10 seconds}
  \item \textbf{Query Processing:} Natural language to visualization in \textbf{\(\sim\)2--4 seconds}
  \item \textbf{Data Capacity:} Handles datasets up to \textbf{200MB} (CSV limit)
  \item \textbf{Tool Selection Accuracy:} \textbf{\(\sim\)95\%} correct analysis method selection
  \item \textbf{Visualization Types:} \textbf{12+ chart types} with automatic selection
  \item \textbf{Fallback System:} Automatic model switching on rate limits (Pro \(\rightarrow\) Flash)
\end{itemize}

\section*{Core Capabilities \& Analysis Tools}

\subsection*{Statistical Analysis}
\begin{itemize}
  \item Comprehensive dataset summaries and statistical overviews
  \item Correlation analysis with heatmap generation
  \item Group-by aggregations (sum, mean, count, max, min)
  \item Distribution analysis via histograms and box plots
\end{itemize}

\subsection*{Advanced Analytics}
\begin{itemize}
  \item \textbf{Time Series Decomposition} --- Trend, seasonal, and residual analysis
  \item \textbf{Outlier Detection} --- IQR and Z-score based anomaly identification
  \item \textbf{Feature Importance} --- Random Forest-based predictive modeling
  \item \textbf{Distribution Comparison} --- Violin plots for cross-category analysis
\end{itemize}

\subsection*{Smart Visualization Engine}
\begin{itemize}
  \item Automatic chart type selection based on data characteristics
  \item 7 color palettes with force override capability
  \item Interactive Plotly charts with export functionality
  \item Responsive layout with adjustable columns and theming
\end{itemize}

\section*{Visual Results \& Interface Demonstrations}

\begin{figure}[h!]
  \centering
  \includegraphics[width=\linewidth]{Image_1.jpg}
  \caption{Auto Dashboard Interface --- Main dashboard showing the ``Intelligent Dashboard'' with uploaded \texttt{car\_prices.csv}; sidebar controls for preprocessing and customization on the left.}
\end{figure}

\begin{figure}[h!]
  \centering
  \includegraphics[width=\linewidth]{Image_2.jpg}
  \caption{Dashboard Management --- Drag-to-reorder interface with 8 active plots (summary, histograms, correlation heatmap, scatter, distribution comparisons). Draggable elements highlighted for intuitive reordering.}
\end{figure}

\begin{figure}[h!]
  \centering
  \includegraphics[width=\linewidth]{Image_3.jpg}
  \caption{Generated Visualizations --- Auto-generated dashboard with 6 key plots: histogram of odometer, correlation heatmap, scatter (odometer vs. selling price), top car makes bar chart, distribution by body type, and time-series trend analysis.}
\end{figure}

\begin{figure}[h!]
  \centering
  \includegraphics[width=\linewidth]{Image_4.jpg}
  \caption{Custom Plot Generation --- Natural language input (e.g., ``Show sales trends over time'') with prompt suggestions such as revenue comparison, customer age distribution, and feature correlation analysis.}
\end{figure}

\begin{figure}[h!]
  \centering
  \includegraphics[width=\linewidth]{Image_5.jpg}
  \caption{Natural Language Query Results --- Query ``What are the top 5 categories by revenue?'' processed with Gemini 2.5 Pro, returning an interactive bar chart (e.g., Ford leading with \$1.36B).}
\end{figure}

\begin{figure}[h!]
  \centering
  \includegraphics[width=\linewidth]{Image_6.jpg}
  \caption{Data Preview \& Schema --- First 10 rows of the car dataset with schema details (types: \texttt{object}, \texttt{int64}, \texttt{float64}) for columns such as year, make, model, body type, transmission, VIN, state, condition, odometer, color, seller, MMR, selling price, and sale date.}
\end{figure}

\begin{figure}[h!]
  \centering
  \includegraphics[width=\linewidth]{Image_7.jpg}
  \caption{Data Quality Report --- Example: 558{,}837 rows \(\times\) 16 columns; 13/16 columns with missing data (transmission 11.69\%, body 2.36\%, condition 2.12\%, trim 1.91\%, model 1.86\%). Types: 11 \texttt{object}, 4 \texttt{float64}.}
\end{figure}

\clearpage

\section*{Implementation Architecture}

\subsection*{Technical Stack}
\begin{itemize}
  \item \textbf{Frontend:} Streamlit with \texttt{streamlit-sortables} for interactive UI
  \item \textbf{AI Models:} Google Gemini 2.5 Pro/Flash via LangChain
  \item \textbf{Data Processing:} Pandas, NumPy, SciPy
  \item \textbf{Visualization:} Plotly for interactive charts
  \item \textbf{ML Libraries:} scikit-learn for advanced analytics
  \item \textbf{Statistical Analysis:} statsmodels for time series decomposition
\end{itemize}

\subsection*{System Components}
\begin{itemize}
  \item \texttt{app.py}: Main Streamlit interface with 4 interactive tabs
  \item \texttt{tools.py}: 15+ analysis functions with robust error handling
  \item \texttt{chains.py}: LangChain integration for AI tool binding
  \item \texttt{system\_analyst.md}: Expert prompt engineering for data analyst behavior
\end{itemize}

\section*{Key Innovations}

\subsection*{Intelligent Features}
\begin{itemize}
  \item \textbf{Smart Column Matching} --- Fuzzy matching for column name suggestions
  \item \textbf{Type Coercion} --- Automatic conversion of string numbers (\$, \%, commas)
  \item \textbf{Error Recovery} --- Helpful suggestions when operations fail
  \item \textbf{Memory Optimization} --- Intelligent sampling for large datasets
  \item \textbf{Business Insights} --- Converts statistical findings to actionable language
\end{itemize}

\section*{Deployment \& Configuration}

\subsection*{Quick Setup}
\begin{verbatim}
# API Configuration
GOOGLE_AI_API_KEY=your_api_key

# Model Selection (configurable)
Primary Model: gemini-2.5-pro
Fallback Model: gemini-2.5-flash
\end{verbatim}

\section*{Use Cases \& Impact}

\subsection*{Business Intelligence}
\begin{itemize}
  \item Democratizes data analysis for non-technical users
  \item Reduces time-to-insight from hours to seconds
  \item Enables self-service analytics without SQL knowledge
  \item Generates executive-ready reports and dashboards
\end{itemize}

\subsection*{Target Users}
\begin{itemize}
  \item Business analysts seeking rapid insights
  \item Product managers needing quick data exploration
  \item Executives requiring automated reporting
  \item Data teams looking to accelerate analysis workflows
\end{itemize}

\section*{Tech Stack}
Python, Streamlit, LangChain, Google Gemini AI, Plotly, Pandas, Scikit-learn, statsmodels

\vspace{1em}
\emph{This platform represents a paradigm shift in data analysis accessibility, combining the power of large language models with sophisticated data science techniques to create an intuitive, yet powerful analytics solution that bridges the gap between complex data and actionable insights.}

\end{document}


---

### 📊 Inflation Analysis and Interactive Visualization — *Survey-Based Data Analytics Project*  

**Description:**

End-to-end data analytics project analyzing the perceptions and impacts of inflation in Morocco. It combines targeted data collection, EDA, statistical hypothesis testing, and a Flask-based web app with dynamic visualization and prediction features.

**Key Features:**

- Targeted data collection via a custom online form
- Exploratory Data Analysis (distribution plots, correlation matrix, outlier checks)
- Statistical hypothesis testing with real-time result display in the app
- Interactive Flask app: filterable data table, graph gallery, form submission, and user-specific predictions

**Dataset & Collection:**

- **Audience:** students, faculty, professionals (targeted outreach via university emails, LinkedIn, and relevant Facebook groups)
- **Duration:** ~2 weeks of collection
- **Size:** 163 valid responses
- **Privacy-first survey design:** concise form; sensitive questions (e.g., income) handled transparently with confidentiality assurances.

**Methodology:**

**Survey & Cleaning**

- Minimal, focused questionnaire designed for clarity and privacy
- Data cleaning pipeline:
    - Handle missing values & duplicates
    - Detect/treat outliers
    - Normalize/standardize where appropriate
    - Encode categorical variables
    - Custom parser to extract numeric values from income/expense text (e.g., strings with currency hints like “d”, “e”), falling back safely when no number is present.

**Exploratory Data Analysis (EDA)**

- Descriptive stats & distributions (histograms/boxplots)
- Correlation analysis (heatmaps; targeted pairwise checks)
- Guided questions, e.g.:
    - Debt ↔ savings estimation
    - Cost-of-living change ↔ income–expense gap
    - Financial satisfaction ↔ credit presence
    - Household size ↔ estimated expenses
    - Occupation (student/employee) ↔ perception of price change
    - Demographic effects (age/gender) on satisfaction/savings

**Statistical Hypothesis Testing**

- **Means:** savings estimation differences between genders (CI + test)
- **Independence (χ²):** satisfaction vs. age groups
- **Proportions:** perception of change by student vs. employee (CI + test)
- **One-sample mean:** average income vs. minimum wage (CI + test)
- **One-sample proportion:** change in price perception (CI + test)

**Tech stack:** Python, Flask, Pandas, Plotly, Machine Learning  


**Screenshots:**  

![App Home](./Image2.jpg)  
*Home page of the Inflation Analysis app — entry point to access dataset, forms, and graphs.*  

![Data Collection Form](./Image6.jpg)  
*Inflation Data Collection Form — collecting key socio-economic data from users.*  

![Submission Confirmation](./Image8.jpg)  
*Form submission confirmation page with navigation to table, graphs, and prediction features.*  

![Prediction Result](./Image9.jpg)  
*Prediction interface — provides a summary of personalized insights based on user responses.*  

![Data Visualization](./Image12.jpg)  
*Interactive dashboard — displaying correlation between financial satisfaction and age group.*  

![Hypothesis Test Result](./Image15.jpg)  
*Hypothesis testing popup with statistical results on perceived price change impact.*  


---

### 📝 AI-Generated Text Detection — *NLP & Machine Learning Project*  

**Description:**  
Developed a robust system for detecting AI-generated text using advanced NLP techniques and Machine Learning models.  
The project pipeline integrates data preprocessing, lexical and syntactic feature engineering, and model evaluation on balanced datasets containing both human-written and AI-generated text samples.  

**Key Features:**  
- Comprehensive text preprocessing (cleaning, lemmatization, stopword removal)  
- Feature engineering including word count, average word length, vocabulary richness, POS tags, sentiment analysis, and readability metrics  
- TF-IDF vectorization for contextual feature extraction  
- Comparative study of ML models: Logistic Regression, Naive Bayes, Random Forest, Neural Networks  
- Achieved **99% accuracy** with the Neural Network model on the validation set  

**Modeling Approach:**  
- Baseline models: Logistic Regression, Naive Bayes, Random Forest  
- Deep Learning model: Neural Network with Dropout & ReLU activations  
- Evaluation using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix  


---

**Results Highlights:**  
- Neural Network Model: **99% Accuracy, 99% F1-Score**  
- Logistic Regression: **97.25% Accuracy**  
- Naive Bayes: **91.8% Accuracy**  
- Random Forest: **91.3% Accuracy**  

**Tech stack:** Python, Scikit-learn, NLTK, Pandas, NumPy, TensorFlow/Keras  

---

### 😊 Facial Emotion Recognition with EfficientNetV2M and Attention Mechanisms — *AffectNet Case Study*  

**Description:**  
Developed a deep learning model for **Facial Emotion Recognition (FER)** based on **EfficientNetV2M** architecture, enhanced with attention mechanisms for improved feature focus and model interpretability.  
The system classifies four primary emotions — **Happiness, Sadness, Fear, and Anger** — using the large-scale AffectNet dataset.  

**Key Features:**  
- Fine-tuned **EfficientNetV2M** on AffectNet with integrated attention modules  
- Applied regularization techniques (dropout, batch normalization) to enhance model stability  
- Designed multi-output architecture with **early exits** for adaptable inference speed  
- Achieved **79.3% validation accuracy** after fine-tuning with attention  
- Addressed interpretability through visual attention maps on facial regions  

**Technical Approach:**  
- Transfer learning using ImageNet-pretrained EfficientNetV2M  
- Attention-based feature refinement  
- Early exits for intermediate classification with dynamic inference  
- Hyperparameter tuning with learning rate scheduling and early stopping  
- Categorical cross-entropy loss with per-output tracking  

**Dataset:**  
- AffectNet (subset of 4 emotions)  
- Training set: 80%, Validation set: 20%, Test set: Hold-out  
- Managed class imbalance through targeted data sampling  

**Evaluation Metrics:**  
- Accuracy, F1-Score, Loss Curves  
- Performance validated on unseen AffectNet samples  

**Results Highlights:**  
- Training Accuracy: **96.1%**  
- Validation Accuracy: **79.3%**  
- Validation Loss: **0.53** at best epoch

**Tech stack:** Python, TensorFlow/Keras, EfficientNetV2M, AffectNet Dataset  



**Sample Prediction Result:**  

![FER Prediction Example](./fer_prediction_result.png)  
*Example of a facial emotion recognition prediction on unseen data — the model correctly identified the emotion based on key facial features.*  


---

### 🔤 English ↔ Darija (Moroccan Arabic) Neural Machine Translation 

**Description:** Built a low-resource English–Darija translation pipeline: cleaned a noisy parallel corpus, filled missing English sentences via back-translation, then trained a custom seq2seq LSTM (with a peephole variant) and ran a small hyper-parameter study.

**Problem:** Public Darija–English datasets are small and very incomplete (≈85% of the eng side missing in the sentences split). Dropping those rows would destroy supervision, so we first recover the English side before modeling.

**Data:** Hugging Face imomayiz/darija-english (config: sentences) with darija (Latin), darija_ar (Arabic script), and eng; most eng entries are missing.

**Approach:** EDA + missing-value audit → back-translation of darija_ar using ychafiqui/darija-to-english-2 ( Transformers) with batching & checkpointing; split the missing set into 5 parts for parallel runs, then recombined.

**Preprocessing:** Lowercasing + punctuation strip; word-level tokenization (NLTK); vocab with special tokens (<PAD>, <SOS>, <EOS>, <UNK>); index + pad; build train/val/test loaders (80/10/10).

**Models:** Baseline seq2seq LSTM encoder–decoder (PyTorch) and an advanced variant with a custom LSTM Peephole cell; Seq2Seq forward with teacher forcing; metrics: token-level cross-entropy and validation accuracy.

**Tech stack:** PyTorch, Transformers (HF), datasets (HF), NLTK, pandas, tqdm, matplotlib.


---


### 🔒 Advanced Data Security for Healthcare Systems — *Oracle Database Administration Project*  

**Description:**  
Designed and implemented a secure database architecture for healthcare data management using **Oracle Database**.  
This project integrates advanced security mechanisms such as encryption, masking, role-based access control, and auditing to ensure data confidentiality, integrity, and compliance with healthcare regulations.  

**Key Features:**  
- Role-based access control with user-specific privileges  
- Data encryption for sensitive patient, medical, and financial information  
- Data masking on critical fields (personal details, diagnoses, contact information)  
- Database Vault for enhanced data access control  
- Unified auditing for activity monitoring and anomaly detection  
- Secure data model covering patients, doctors, hospitals, diagnostics, and billing  

**Core Entities & Relationships:**  
- Patient, Doctor, Nurse, Hospital, Bill, Patient_Diagnostic  
- Managed via one-to-many and many-to-one relationships with enforced referential integrity  

**Technical Approach:**  
- Oracle SQL Developer for schema design and user management  
- Implementation of encryption wallets and key management  
- Data masking with Oracle Data Redaction  
- Unified Auditing for centralized security monitoring  

**Tech stack:** Oracle Database, SQL Developer, Oracle Data Vault, Oracle Unified Auditing  


---


### 🌦️ IoT-Based Smart Weather Detection System — *SOLLIS Project*  

**Description:**  
Developed a smart weather detection system leveraging IoT technologies for real-time indoor environment monitoring.  
The system integrates sensors, actuators, and data visualization dashboards, enabling dynamic control and monitoring of temperature, humidity, and light intensity within indoor spaces.  

**Key Features:**  
- Real-time data collection from light and temperature sensors  
- Adaptive LED brightness control based on ambient light levels  
- LCD display showing live temperature and weather status  
- Raspberry Pi-based gateway handling data processing and MQTT communication  
- Node-RED dashboard for real-time visualization and historical trend analysis  
- Continuous data updates and user interaction with the environment  

**System Components:**  
- Arduino + Sensors (Photoresistor, DHT22)  
- Raspberry Pi (data gateway & MQTT broker)  
- Node-RED Dashboard (UI & data visualization)  
- Actuators: LEDs, LCD Screen  

**Example Workflow:**  
- The system detects light intensity and temperature  
- LEDs adjust brightness based on detected light  
- Data sent via MQTT to Node-RED for visualization  
- User views real-time and historical data on the dashboard  

**Tech stack:** Arduino, Raspberry Pi, Node-RED, Python, MQTT  

**Screenshots:**  

![Complete System Setup](./Image17.jpg)  
*Complete system setup integrating Arduino with sensors, actuators, and Raspberry Pi gateway — demonstrating full IoT workflow and communication setup.*  

![LCD Display Output](./Image18.jpg)  
*LCD display showing real-time weather status and temperature readings, providing direct feedback from the sensors.*  

![Sensor and Actuator Integration](./Image19.jpg)  
*Close-up of the sensor and actuator setup on a breadboard — showcasing LEDs reacting to ambient light and live data being processed by the Arduino.*  

![Node-RED Flow Diagram](./Image20.png)  
*Node-RED flow representing the system’s logic — managing communication between the gateway, sensors, actuators, and the user interface. It handles light threshold checks, command routing, and dashboard interactions.*  



---

### 🎧 Audio Compression System — *IRM Custom Audio Format*  

![IRM Audio Compression Interface](./Image1.png)  
*Interface of the application allowing users to select audio files and apply the IRM compression algorithm.*  

![Compression Result Example](./image.png)  
*Output showcasing compression results, file size reduction, and comparison with standard formats.*  

**Description:**  
Designed a custom audio compression system named **IRM**, combining **Discrete Wavelet Transform (DWT)** for multi-resolution analysis with the **Lempel-Ziv-Welch (LZW)** algorithm for data reduction.  
The project integrates a user-friendly interface to apply, visualize, and compare compression results with common formats like WAV and OGG.  

**Key Features:**  
- Multi-resolution signal analysis with DWT  
- Entropy-based data compression using LZW  
- Compression/decompression process with quality retention  
- GUI-based interaction for audio processing  

**Tech stack:** Python, NumPy, Tkinter  

---


### 🏥 Healthcare Data Processing Pipeline  
**Description:** Built a real-time Big Data pipeline for healthcare data using Pulsar, Airflow, TensorFlow, and Neo4j.  
**Technologies:** Apache Pulsar, Airflow, Snowflake, Neo4j, TensorFlow  


---

## 📊 Data Analysis and Visualization Projects  

- **Revenue Prediction Model:** Supervised ML model for income prediction. *(Python, Scikit-learn)*  
- **Password Strength Classifier:** ML system for evaluating password robustness. *(Python, XGBoost)*  
- **Fake News Detection:** Machine Learning classifier for fake news detection. *(Python, NLP, Scikit-learn)*  
- **Sales Dashboard in Excel:** Sales data analysis with Excel dashboards.  
- **Web Planner App with Calendar:** Django-based planner with category management and interactive calendar. *(Django, FullCalendar)*  
- **Power BI Dashboard - Blinkit Sales Performance:** Visualized Blinkit app sales with Power BI.  



---

### 🍳 Web Application for Sharing Culinary Recipes — *Spring Boot MVC Project*  

**Description:**  
Designed and developed a full-stack web application for culinary recipe sharing.  
The system allows users to explore, publish, edit, and interact with recipes through comments and favorites. The project emphasizes modular design using **Spring Boot (MVC Architecture)**, ensuring scalability, maintainability, and a smooth user experience.  

**Key Features:**  
- User authentication, registration, and profile management  
- Personal dashboard displaying user's recipes and comments  
- CRUD operations for recipe management with image upload support  
- Community interaction through comments and recipe favoriting  
- Categorization of recipes for easy browsing  
- MVC architecture with Thymeleaf front-end  

**Core System Components:**  
- **Entities:** User, Recipe, Comment, Category  
- **Relationships:**  
  - One-to-Many (User–Recipe, User–Comment)  
  - Many-to-Many (User–Favorite Recipes)  
  - One-to-Many (Category–Recipe)  

**Tech stack:** Java, Spring Boot, Thymeleaf, MySQL, Spring Security, JPA/Hibernate  


---

## 💼 Professional Experience  

### 🕵️‍♂️ AI-Based Document Forgery Detection — *Attijariwafa Bank Industrial Project*  

**Description:**  
Developed a hybrid AI system for detecting forged regions in digitized documents by combining **Visual Deep Learning (SegFormer)** with **Semantic Analysis (OCR + LLaMA-3)**.  
The system aims to localize tampered areas with high precision and analyze textual inconsistencies, providing a comprehensive fraud detection solution for banking operations.  

**Key Features:**  
- Visual forgery detection using SegFormer-based segmentation model  
- Hierarchical OCR pipeline for structured text extraction from scanned documents  
- Semantic inconsistency analysis powered by LLaMA-3 large language model  
- Microservice-based architecture with Flask (backend) and React (frontend)  
- Explainable AI with visual attention overlays and semantic mismatch reporting  

**Technical Approach:**  
- Custom dataset construction with tampered and original documents  
- Fine-tuned SegFormer model for binary tampering segmentation with hybrid loss (Focal Tversky + Dice)  
- OCR pipeline combining Tesseract and LLMWhisperer  
- LLaMA-3 based semantic detection of anomalies in extracted text  
- REST APIs serving segmentation and semantic analysis results  

**Key Results:**  
- Visual detection recall: **86.5%** on evaluation set  
- Average IoU on segmentation masks: **0.72**  
- System throughput: **< 1.5s per document** on benchmark hardware

**Tech stack:** Python, PyTorch, SegFormer, OCR (LLMWhisperer,Tesseract), LLaMA-3 (Groq API), MongoDB, Flask, React  



**Sample Results:**  

| Ground Truth vs Predicted Masks |  
|---|  
| ![Case 1 - Ground Truth & Prediction](./RES1.png) |  
*Case 1 — Left: Ground Truth Mask, Right: Model Predicted Mask*  

| ![Case 2 - Ground Truth & Prediction](./RES2.png) |  
*Case 2 — Left: Ground Truth Mask, Right: Model Predicted Mask*  

| ![Case 3 - Ground Truth & Prediction](./RES3.png) |  
*Case 3 — Left: Ground Truth Mask, Right: Model Predicted Mask*  

---

**Visual Tampering Detection Interface:**  

![Tampering Detection on Document Sample](./RES4.png)  
*Tampering visualization in the deployed system — the model successfully highlighted forged areas directly on the scanned document within the web application interface.*  






---

### 🛂 Automatic License Plate Recognition System — *Marsa Maroc Security Project*  

**Description:**  
Developed a real-time **Automatic License Plate Recognition (ALPR)** system using YOLOv5 for securing vehicle exits at Marsa Maroc.  
The system detects license plates, extracts characters with OCR, and verifies against a database of valid exit permits, aiming to automate and enhance port security.  

**Key Features:**  
- Real-time license plate detection with **YOLOv5** (trained for 30 epochs)  
- OCR character extraction with **YOLOv5-based multi-class detection**  
- Video processing pipeline with detection and database validation  
- Django-based web interface (in progress) for real-time monitoring  
- Integrated MySQL for permit management and system logging  

**Performance Highlights:**  
- License Plate Detection Precision: **97.1%** — Recall: **98.5%**  
- OCR Character Recognition Precision: **88%** — Recall: **91.9%**  
- Video Frame Processing Time: **~614ms – 1292ms per frame**  
- Trained on custom dataset with extended augmentation for robustness


**Visual Results & Sample Predictions:**  

![Full Vehicle Detection with License Plate](./Image_car.jpg)  

![Extracted License Plate for OCR](./plate_to_extract_text_from.jpg)  
*Detected license plate cropped from the vehicle image — ready for character recognition.*  

![YOLOv5 OCR Character Detection Result](./extraction_of_text.jpg)  
*Character detection output on the license plate — bounding boxes with confidence scores for each recognized character.*  

![Sorted Character Predictions from Model](./detection_res_car.jpg)  
*Sorted character predictions with bounding box details and class mapping — automated sorting for correct license plate reconstruction.*  

![Class Mapping Dictionary](./mappage.png)  
*Sample of class mapping dictionary used to decode YOLOv5 class IDs into readable characters (digits/letters).*  

![YOLOv5 Real-Time Detection on Video Frame](./video_detection_res.jpg)  
*YOLOv5 running on a live video frame — real-time detection and character recognition results during video processing.*  


**Tech stack:** Python, YOLOv5, OpenCV, MySQL, Django, PhpMyAdmin  

---



### 🛒 E-Commerce Platform Development — *Palfarism*  
**Description:** Developed a full-featured e-commerce platform using PHP and MySQL following MVC architecture.  
**Key Deliverables:**

- **Role-based auth & access control:** Visitors, customers, and admins, each with a dedicated dashboard.
- **Catalog & inventory:** Complete product catalog with category management and CRUD for products/stock via the admin interface.
- **Secure order/payment flow:** Password hashing, robust ID management, and step-by-step order handling.
- **Admin–seller operations dashboard:** Real-time tracking of client info, items, order status, and direct contact actions.
- **Responsive dashboards:** Data-driven views for sales, inventory status, and activity monitoring.

**Impact:** Digitized the sales cycle and strengthened the client’s online presence, opening a new revenue channel.

**Tech stack:** PHP, MySQL, HTML, CSS, Bootstrap  

---

## 📝 Contact  
Feel free to reach out for collaboration or inquiries!  

---

