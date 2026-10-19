# =============================================================================
#  MAKINI CALEB — Live Analytics Dashboard 
# =============================================================================

need <- c("shiny","shinydashboard","DT","ggplot2","dplyr","tidyr",
          "caret","randomForest","e1071","xgboost","forecast","tseries",
          "cluster","factoextra","pROC","corrplot","gridExtra","grid",
          "scales","class","FNN","Metrics","GGally","ggridges","reshape2")
for (p in need) if (!requireNamespace(p, quietly = TRUE))
  install.packages(p, repos = "https://cloud.r-project.org")
suppressPackageStartupMessages({
  library(shiny); library(shinydashboard); library(DT)
  library(ggplot2); library(dplyr); library(tidyr)
  library(caret); library(randomForest); library(e1071); library(xgboost)
  library(forecast); library(tseries)
  library(cluster); library(factoextra); library(pROC)
  library(corrplot); library(gridExtra); library(grid)
  library(scales); library(class); library(FNN); library(Metrics)
  library(GGally); library(ggridges); library(reshape2)
})
options(shiny.maxRequestSize = 500 * 1024^2)
`%||%` <- function(a,b) if (is.null(a) || length(a) == 0) b else a

# =============================================================================
#  DATA GENERATORS  
# =============================================================================
gen_dtcln <- function() {
  set.seed(1); n <- 300
  d <- data.frame(
    Age = sample(18:75, n, TRUE),
    Income_KES = round(rnorm(n, 55000, 20000)),
    Household_Size = sample(1:10, n, TRUE),
    Farm_Size_Acres = round(runif(n, 0.5, 15), 1),
    Monthly_Expenses_KES = round(rnorm(n, 30000, 8000)),
    Years_Experience = sample(0:40, n, TRUE),
    Education = sample(c("Primary","Secondary","Degree","Masters"), n, TRUE),
    Gender = sample(c("Male","Female"), n, TRUE),
    Extension_Access = sample(c("Yes","No"), n, TRUE),
    Adoption_Status = sample(c("Adopted","Not Adopted"), n, TRUE))
  for (v in c("Age","Income_KES","Farm_Size_Acres")) d[[v]][sample(n,15)] <- NA
  for (v in c("Education","Gender")) d[[v]][sample(n,8)] <- NA
  d$Income_KES[sample(n,5)] <- 500000
  d$Farm_Size_Acres[sample(n,3)] <- 95
  d
}
gen_gss <- function() {
  set.seed(2); n <- 500
  data.frame(
    marital = sample(c("MARRIED","DIVORCED","NEVER MARRIED","WIDOWED"), n, TRUE),
    degree = sample(c("BACHELOR","GRADUATE","JUNIOR COLLEGE","HIGH SCHOOL"), n, TRUE),
    sex = sample(c("MALE","FEMALE"), n, TRUE),
    income = round(rnorm(n, 45000, 20000)),
    age = sample(18:80, n, TRUE),
    hrswrk = sample(10:70, n, TRUE),
    height = round(rnorm(n, 170, 8)),
    weight = round(rnorm(n, 75, 12)))
}
gen_students <- function() {
  set.seed(7); n <- 300
  study <- round(rnorm(n, 14, 4), 1)
  data.frame(
    Student_ID = 1:n,
    Gender = sample(c("Male","Female","Non-binary"), n, TRUE),
    Major = sample(c("STEM","Business","Humanities"), n, TRUE),
    Study_Hours = study,
    GPA = round(pmin(4, pmax(0, 1.85 + 0.086*study + rnorm(n,0,.25))), 2),
    Pre_Test_Score = round(rnorm(n, 60, 10)),
    Post_Test_Score = round(rnorm(n, 68.6, 10)),
    Passed_Class = ifelse(runif(n) > 0.25, "Yes", "No"))
}
gen_employees <- function() {
  set.seed(11); n <- 300
  exp <- sample(0:25, n, TRUE); train <- round(runif(n, 5, 60))
  stress <- pmin(10, pmax(1, 9 - 0.15*exp + rnorm(n, 0, 1.2)))
  prod <- 40 + 0.5*train + 0.4*exp - 1.2*stress + rnorm(n, 0, 3)
  data.frame(
    ID = 1:n,
    Department = sample(c("Sales","Engineering","HR"), n, TRUE),
    Work_Mode = sample(c("Remote","Hybrid","Onsite"), n, TRUE),
    Years_Experience = exp,
    Training_Hours = train,
    Stress_Index = round(stress, 2),
    Productivity_Score = round(prod, 2))
}
gen_ts <- function() {
  m <- 48
  data.frame(Temperature = round(25 + 8*sin(seq(0, 3*2*pi, length.out=m)) +
                                   rnorm(m, 0, 2), 2))
}
gen_mall <- function() {
  set.seed(21); n <- 400
  data.frame(
    CustomerID = 1:n,
    Age = sample(18:70, n, TRUE),
    Annual_Income = round(rnorm(n, 60000, 20000)),
    Spending_Score = sample(1:100, n, TRUE),
    Average_Purchase_Value = round(runif(n, 20, 500), 2))
}
gen_pca <- function() {
  set.seed(5); n <- 300
  x1 <- rnorm(n); x2 <- 0.9*x1 + rnorm(n,0,.3); x3 <- -0.7*x1 + rnorm(n,0,.5)
  x4 <- rnorm(n); x5 <- -0.8*x4 + rnorm(n,0,.4); x6 <- rnorm(n)
  x7 <- 0.6*x6 + rnorm(n,0,.5); x8 <- rnorm(n); x9 <- 0.7*x8 + rnorm(n,0,.4)
  x10 <- -0.5*x1 + rnorm(n,0,.5)
  as.data.frame(scale(cbind(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10)))
}
gen_loan <- function() {
  set.seed(9); n <- 600
  inc <- sample(c(runif(240, 17000, 42000), runif(360, 44000, 135000)), n)
  data.frame(
    Age = sample(21:65, n, TRUE),
    Income = round(inc),
    Credit_Score = round(rnorm(n, 650, 100)),
    Loan_Amount = round(runif(n, 1000, 50000)),
    Employment_Years = sample(0:30, n, TRUE),
    Loan_Approved = factor(ifelse(inc > 43000, "Yes", "No"), levels = c("No","Yes")))
}
gen_diabetes <- function() {
  set.seed(3); n <- 1000
  hba1c <- round(rnorm(n, 5.5, 1.2), 2)
  glucose <- round(rnorm(n, 140, 40), 1)
  data.frame(
    age = round(runif(n, 1, 80)),
    hypertension = factor(sample(c("No","Yes"), n, TRUE, prob = c(0.9, 0.1)),
                          levels = c("No","Yes")),
    heart_disease = factor(sample(c("No","Yes"), n, TRUE, prob = c(0.95, 0.05)),
                           levels = c("No","Yes")),
    smoking_history = factor(sample(c("never","former","current",
                                      "not current","ever","No Info"), n, TRUE)),
    BMI = round(rnorm(n, 27, 6), 2),
    HbA1c_level = hba1c,
    blood_glucose_level = glucose,
    diabetes = factor(ifelse(hba1c > 6.5 & glucose > 180, "Yes", "No"),
                      levels = c("No","Yes")))
}
gen_eco <- function() {
  set.seed(4); n <- 500
  txn <- sample(seq.Date(as.Date("2023-01-01"), as.Date("2023-12-31"), by="day"), n, TRUE)
  qty <- sample(1:5, n, TRUE); pr <- round(runif(n, 20, 500), 2)
  d <- data.frame(
    TransactionID = 1:n, CustomerID = sample(1:150, n, TRUE),
    Quantity = qty,
    PaymentMethod = sample(c("Credit Card","Debit Card","UPI","Cash on Delivery"), n, TRUE),
    TransactionDate = txn,
    ProductCategory = sample(c("Clothing","Electronics","Home Appliances"), n, TRUE,
                             prob = c(0.4, 0.35, 0.25)),
    Price = pr, Rating = round(runif(n, 1, 5), 1),
    TotalAmount = round(pr * qty, 2),
    Age = sample(18:70, n, TRUE),
    Gender = sample(c("Male","Female"), n, TRUE),
    Location = sample(c("Nairobi","Mombasa","Kisumu","Nakuru","Eldoret"), n, TRUE),
    MembershipStatus = sample(c("Basic","Silver","Gold","Premium"), n, TRUE))
  d$Age[sample(n, 20)] <- NA
  d
}

# =============================================================================
#  UI  
# =============================================================================
ui <- dashboardPage(
  skin = "purple",
  dashboardHeader(
    title = tags$span(icon("chart-line"), " MAKINI CALEB"),
    titleWidth = 300,
    tags$li(class = "dropdown",
            tags$a(href = "#", icon("graduation-cap"), " BSc Mathematics")),
    tags$li(class = "dropdown",
            tags$a(href = "#", icon("database"), " Live Analytics"))),
  dashboardSidebar(width = 285, sidebarMenu(id = "tabs",
                                            menuItem("🏠 Home",               tabName = "home",   icon = icon("house")),
                                            menuItem("🧹 Data Cleaning",       tabName = "clean",  icon = icon("broom")),
                                            menuItem("🔎 Wrangling & Joins",   tabName = "wrang",  icon = icon("filter")),
                                            menuItem("📊 Visualization",       tabName = "viz",    icon = icon("chart-column")),
                                            menuItem("🧪 Statistics & Reg",    tabName = "stat",   icon = icon("flask")),
                                            menuItem("⏳ Time Series",         tabName = "ts",     icon = icon("clock")),
                                            menuItem("🔵 K-Means & PCA",       tabName = "unsup",  icon = icon("circle-nodes")),
                                            menuItem("🤖 KNN · Logistic · SVM",tabName = "sup",    icon = icon("robot")),
                                            menuItem("🌲 Random Forest",       tabName = "rf",     icon = icon("tree")),
                                            menuItem("⚡ XGBoost",             tabName = "xgb",    icon = icon("bolt")),
                                            menuItem("💡 Explainable AI",      tabName = "xai",    icon = icon("lightbulb")),
                                            menuItem("🛒 Project — E-commerce",tabName = "proj",   icon = icon("cart-shopping"))
  )),
  dashboardBody(
    tags$head(tags$style(HTML("
      body { font-family: 'Segoe UI', Tahoma, sans-serif; }
      .content-wrapper { background-color: #f4f6fb; }
      .box { border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,.06); }
      .hero { background: linear-gradient(135deg,#6a11cb 0%,#2575fc 100%);
        color:#fff; padding:34px; border-radius:16px; margin-bottom:24px; }
      .hero h1 { font-weight:800; margin:0 0 6px 0; letter-spacing:1px; }
      .hero p { font-size:16px; opacity:.95; margin:0; }
      .badge-soft { display:inline-block; background:rgba(255,255,255,.18);
        color:#fff; border:1px solid rgba(255,255,255,.4);
        padding:4px 12px; border-radius:20px; margin:4px 4px 0 0;
        font-size:12px; font-weight:500; }
      .stat-card { background:#fff; border-radius:12px; padding:20px;
        text-align:center; box-shadow: 0 2px 10px rgba(0,0,0,.06); }
      .stat-card .num { font-size:32px; font-weight:800;
        background: linear-gradient(135deg,#6a11cb,#2575fc);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
      .stat-card .lbl { font-size:13px; color:#666; font-weight:600; }
      .pred-box { background: linear-gradient(135deg,#27ae60,#16a085);
        color:#fff; padding:22px; border-radius:12px; text-align:center;
        box-shadow: 0 4px 14px rgba(39,174,96,.25); }
      .pred-box .val { font-size:34px; font-weight:800; margin:6px 0; }
      .pred-box .lbl { font-size:13px; opacity:.9; text-transform:uppercase;
        letter-spacing:1px; }
      .pred-box.warn { background: linear-gradient(135deg,#e67e22,#d35400); }
      .pred-box.info { background: linear-gradient(135deg,#3498db,#2980b9); }
      h2.section-title { font-weight:800; color:#333; margin-bottom:18px; }
      .form-panel { background:#fafbfc; padding:16px; border-radius:10px;
        border-left:4px solid #6a11cb; }
    "))),
    tabItems(
      
      tabItem(tabName = "home",
              div(class = "hero",
                  h1("MAKINI CALEB — Live Analytics"),
                  p("BSc Mathematics · Data Scientist · Every chart and prediction is generated live in R."),
                  div(span(class="badge-soft","Live Plots"),
                      span(class="badge-soft","New Predictions"),
                      span(class="badge-soft","Forecasts"),
                      span(class="badge-soft","Explainable AI"))),
              fluidRow(
                column(3, div(class="stat-card",
                              div(class="num","12"), div(class="lbl","Analysis Modules"))),
                column(3, div(class="stat-card",
                              div(class="num","30+"), div(class="lbl","Live Charts"))),
                column(3, div(class="stat-card",
                              div(class="num","18"), div(class="lbl","Predictors"))),
                column(3, div(class="stat-card",
                              div(class="num","100%"), div(class="lbl","Runnable")))),
              fluidRow(box(width = 12, status = "primary", solidHeader = TRUE,
                           title = "🎯 Every model has a live prediction form",
                           tags$ul(
                             tags$li(tags$b("🤖 KNN, Logistic, SVM"), " — enter values, get classification"),
                             tags$li(tags$b("🌲 Random Forest"), " — regression (MPG) & classification (Iris)"),
                             tags$li(tags$b("⚡ XGBoost"), " — diabetes prediction with probability"),
                             tags$li(tags$b("💡 XAI"), " — global + local explanation"),
                             tags$li(tags$b("⏳ Time Series"), " — interactive ARIMA forecast"),
                             tags$li(tags$b("🛒 Project"), " — full e-commerce analytics"))))
      ),
      
      tabItem(tabName = "clean",
              h2(class="section-title","🧹 Data Cleaning — Before & After"),
              fluidRow(valueBoxOutput("cl_miss", width=3), valueBoxOutput("cl_outl", width=3),
                       valueBoxOutput("cl_rows", width=3), valueBoxOutput("cl_cols", width=3)),
              fluidRow(
                box(title="NA by Column", status="warning", width=6, solidHeader=TRUE,
                    plotOutput("cl_na", height=320)),
                box(title="Numeric Distributions (After)", status="success",
                    width=6, solidHeader=TRUE, plotOutput("cl_box", height=320))),
              fluidRow(
                box(title="Outliers Detected (IQR)", status="danger",
                    width=6, solidHeader=TRUE, DTOutput("cl_outliers")),
                box(title="Cleaned Data Preview", status="primary",
                    width=6, solidHeader=TRUE, DTOutput("cl_preview")))
      ),
      
      tabItem(tabName = "wrang",
              h2(class="section-title","🔎 Wrangling, Joins & Reshaping"),
              fluidRow(
                box(title="Filter — MARRIED / DIVORCED", status="primary", width=6,
                    solidHeader=TRUE, DTOutput("wr_filt")),
                box(title="Six Join Types", status="info", width=6, solidHeader=TRUE,
                    tabsetPanel(
                      tabPanel("Inner", DTOutput("j_inner")),
                      tabPanel("Left",  DTOutput("j_left")),
                      tabPanel("Right", DTOutput("j_right")),
                      tabPanel("Full",  DTOutput("j_full")),
                      tabPanel("Semi",  DTOutput("j_semi")),
                      tabPanel("Anti",  DTOutput("j_anti"))))),
              fluidRow(
                box(title="Wide → Long", status="success", width=6,
                    solidHeader=TRUE, DTOutput("pv_long")),
                box(title="Long → Wide", status="success", width=6,
                    solidHeader=TRUE, DTOutput("pv_wide")))
      ),
      
      tabItem(tabName = "viz",
              h2(class="section-title","📊 Visualization Gallery"),
              fluidRow(column(3, selectInput("vz_type", "Chart",
                                             c("Histogram","Boxplot","Scatter+Fit","Violin","Ridge",
                                               "Correlation Heatmap","Pairs Plot","Pie")))),
              fluidRow(box(width=12, status="primary", solidHeader=TRUE,
                           title="Live chart", plotOutput("vz_plot", height=520))),
              fluidRow(box(width=12, status="info", solidHeader=TRUE,
                           title="Grouped Summary", DTOutput("vz_summary")))
      ),
      
      tabItem(tabName = "stat",
              h2(class="section-title","🧪 Hypothesis Tests & Regression"),
              fluidRow(
                box(title="Test Output", status="primary", width=6,
                    solidHeader=TRUE, verbatimTextOutput("st_out")),
                box(title="Coefficient Table", status="success", width=6,
                    solidHeader=TRUE, DTOutput("st_coef"))),
              fluidRow(
                box(title="Scatter + Linear Fit", status="info", width=6,
                    solidHeader=TRUE, plotOutput("st_plot")),
                box(title="Diagnostic Plots", status="warning",
                    width=6, solidHeader=TRUE, plotOutput("st_diag"))),
              fluidRow(box(title="ANOVA + Chi-Square", status="danger", width=12,
                           solidHeader=TRUE, verbatimTextOutput("st_anova")))
      ),
      
      tabItem(tabName = "ts",
              h2(class="section-title","⏳ Time Series Analysis"),
              fluidRow(column(3, sliderInput("ts_h", "Forecast horizon (months)",
                                             1, 24, 12))),
              fluidRow(box(title="Series + Forecast", status="primary", width=12,
                           solidHeader=TRUE, plotOutput("ts_plot"))),
              fluidRow(
                box(title="Decomposition", status="success", width=6,
                    solidHeader=TRUE, plotOutput("ts_dec", height=400)),
                box(title="ACF / PACF", status="info", width=6,
                    solidHeader=TRUE, plotOutput("ts_acf", height=400))),
              fluidRow(
                box(title="ADF Test", status="warning", width=4,
                    solidHeader=TRUE, verbatimTextOutput("ts_adf")),
                box(title="Forecast Values", status="danger", width=8,
                    solidHeader=TRUE, DTOutput("ts_forecast_tbl")))
      ),
      
      tabItem(tabName = "unsup",
              h2(class="section-title","🔵 K-Means Clustering & PCA"),
              fluidRow(
                box(title="Elbow Method", status="primary", width=6,
                    solidHeader=TRUE, plotOutput("km_elbow")),
                box(title="Cluster Visualisation (k = 3)", status="success",
                    width=6, solidHeader=TRUE, plotOutput("km_plot"))),
              fluidRow(box(title="Cluster Summary", status="info", width=12,
                           solidHeader=TRUE, DTOutput("km_summary"))),
              fluidRow(
                box(title="PCA Scree", status="warning", width=6,
                    solidHeader=TRUE, plotOutput("pc_scree")),
                box(title="PCA Biplot", status="primary", width=6,
                    solidHeader=TRUE, plotOutput("pc_biplot"))),
              fluidRow(
                box(title="Variance Table", status="info", width=6,
                    solidHeader=TRUE, DTOutput("pc_var")),
                box(title="Reduced Data (First 4 PCs)", status="success",
                    width=6, solidHeader=TRUE, DTOutput("pc_red")))
      ),
      
      tabItem(tabName = "sup",
              h2(class="section-title","🤖 KNN, Logistic Regression & SVM"),
              
              fluidRow(
                box(title="🎯 KNN — Predict New Customer", status="primary",
                    width=4, solidHeader=TRUE,
                    div(class="form-panel",
                        numericInput("kn_age",  "Age", 35, 21, 65, 1),
                        numericInput("kn_inc",  "Income (KES)", 85000, 15000, 150000, 1000),
                        numericInput("kn_loan", "Loan Amount (KES)", 20000, 1000, 50000, 500),
                        numericInput("kn_cs",   "Credit Score", 700, 300, 850, 1),
                        helpText(HTML("<small>Income &gt; 43,000 → likely <b>Yes</b></small>")),
                        actionButton("kn_predict", "Predict", icon = icon("wand-magic-sparkles"),
                                     class = "btn-success", width = "100%"))),
                box(title="Prediction", status="success", width=4,
                    solidHeader=TRUE, uiOutput("kn_result")),
                box(title="KNN — Accuracy vs K", status="info", width=4,
                    solidHeader=TRUE, plotOutput("kn_acc", height=300))),
              fluidRow(
                box(title="KNN — Confusion Matrix (k = 5)", status="warning",
                    width=6, solidHeader=TRUE, plotOutput("kn_cm", height=340)),
                box(title="KNN — Test Predictions", status="primary", width=6,
                    solidHeader=TRUE, DTOutput("kn_pred"))),
              
              fluidRow(
                box(title="🎯 Logistic — Predict Diabetes", status="danger",
                    width=4, solidHeader=TRUE,
                    div(class="form-panel",
                        numericInput("lg_age", "Age", 45, 1, 100),
                        selectInput("lg_htn", "Hypertension", c("No","Yes")),
                        selectInput("lg_hd",  "Heart Disease", c("No","Yes")),
                        numericInput("lg_bmi", "BMI", 27, 10, 60),
                        numericInput("lg_hba", "HbA1c", 5.5, 3, 12, 0.1),
                        numericInput("lg_glu", "Blood Glucose", 140, 50, 400),
                        actionButton("lg_predict", "Predict", icon = icon("wand-magic-sparkles"),
                                     class = "btn-danger", width = "100%"))),
                box(title="Prediction", status="danger", width=4,
                    solidHeader=TRUE, uiOutput("lg_result")),
                box(title="Logistic — ROC", status="primary", width=4,
                    solidHeader=TRUE, plotOutput("lg_roc", height=300))),
              fluidRow(
                box(title="Coefficients", status="info", width=6,
                    solidHeader=TRUE, DTOutput("lg_coef")),
                box(title="Confusion Matrix", status="warning", width=6,
                    solidHeader=TRUE, plotOutput("lg_cm", height=340))),
              
              fluidRow(
                box(title="🎯 SVM — Classify Iris Flower", status="success",
                    width=4, solidHeader=TRUE,
                    div(class="form-panel",
                        numericInput("sv_sl", "Sepal Length", 5.1, 4, 8, 0.1),
                        numericInput("sv_sw", "Sepal Width",  3.5, 2, 5, 0.1),
                        numericInput("sv_pl", "Petal Length", 1.4, 1, 7, 0.1),
                        numericInput("sv_pw", "Petal Width",  0.2, 0.1, 2.5, 0.1),
                        actionButton("sv_predict", "Predict", icon = icon("wand-magic-sparkles"),
                                     class = "btn-success", width = "100%"))),
                box(title="Prediction", status="success", width=4,
                    solidHeader=TRUE, uiOutput("sv_result")),
                box(title="SVM — Confusion Matrix", status="primary", width=4,
                    solidHeader=TRUE, plotOutput("sv_cm", height=300)))
      ),
      
      tabItem(tabName = "rf",
              h2(class="section-title","🌲 Random Forest — Regression & Classification"),
              
              fluidRow(
                box(title="🎯 RF Regression — Predict MPG", status="primary",
                    width=4, solidHeader=TRUE,
                    div(class="form-panel",
                        numericInput("rf_cyl",  "Cylinders", 4, 4, 8),
                        numericInput("rf_disp", "Displacement", 120, 50, 500),
                        numericInput("rf_hp",   "Horsepower", 120, 40, 400),
                        numericInput("rf_drat", "Drat", 3.5, 2.5, 5, 0.01),
                        numericInput("rf_wt",   "Weight", 2.5, 1, 6, 0.01),
                        numericInput("rf_qsec", "1/4 mile time", 18, 14, 25, 0.1),
                        numericInput("rf_vs",   "V/S (0/1)", 0, 0, 1),
                        numericInput("rf_am",   "Transmission (0/1)", 0, 0, 1),
                        numericInput("rf_gear", "Gears", 4, 3, 5),
                        numericInput("rf_carb", "Carburettors", 2, 1, 8),
                        actionButton("rf_predict", "Predict MPG", icon = icon("car"),
                                     class = "btn-primary", width = "100%"))),
                box(title="Predicted MPG", status="success", width=4,
                    solidHeader=TRUE, uiOutput("rf_result")),
                box(title="Feature Importance", status="info", width=4,
                    solidHeader=TRUE, plotOutput("rf_imp", height=320))),
              fluidRow(
                box(title="Actual vs Predicted (Test Set)", status="success",
                    width=6, solidHeader=TRUE, plotOutput("rf_act", height=340)),
                box(title="Metrics", status="warning", width=6,
                    solidHeader=TRUE, DTOutput("rf_met"))),
              
              fluidRow(
                box(title="🎯 RF Classification — Predict Iris Species",
                    status="success", width=4, solidHeader=TRUE,
                    div(class="form-panel",
                        numericInput("rfc_sl", "Sepal Length", 5.1, 4, 8, 0.1),
                        numericInput("rfc_sw", "Sepal Width",  3.5, 2, 5, 0.1),
                        numericInput("rfc_pl", "Petal Length", 1.4, 1, 7, 0.1),
                        numericInput("rfc_pw", "Petal Width",  0.2, 0.1, 2.5, 0.1),
                        actionButton("rfc_predict", "Predict Species",
                                     icon = icon("seedling"),
                                     class = "btn-success", width = "100%"))),
                box(title="Predicted Species", status="primary", width=4,
                    solidHeader=TRUE, uiOutput("rfc_result")),
                box(title="Confusion Matrix (Iris)", status="warning",
                    width=4, solidHeader=TRUE, plotOutput("rf_iris", height=320)))
      ),
      
      tabItem(tabName = "xgb",
              h2(class="section-title","⚡ XGBoost — Diabetes Classification"),
              fluidRow(
                box(title="🎯 Predict Diabetes with XGBoost", status="primary",
                    width=4, solidHeader=TRUE,
                    div(class="form-panel",
                        numericInput("xg_age", "Age", 45, 1, 100),
                        selectInput("xg_htn", "Hypertension", c("No","Yes")),
                        selectInput("xg_hd",  "Heart Disease", c("No","Yes")),
                        selectInput("xg_smoke", "Smoking History",
                                    c("never","former","current","not current","ever","No Info")),
                        numericInput("xg_bmi", "BMI", 27, 10, 60),
                        numericInput("xg_hba", "HbA1c", 5.5, 3, 12, 0.1),
                        numericInput("xg_glu", "Blood Glucose", 140, 50, 400),
                        actionButton("xg_predict", "Predict", icon = icon("bolt"),
                                     class = "btn-warning", width = "100%"))),
                box(title="Prediction", status="warning", width=4,
                    solidHeader=TRUE, uiOutput("xg_result")),
                box(title="ROC Curve", status="success", width=4,
                    solidHeader=TRUE, plotOutput("xg_roc", height=320))),
              fluidRow(
                box(title="Feature Importance", status="info", width=6,
                    solidHeader=TRUE, plotOutput("xg_imp", height=340)),
                box(title="Metrics", status="primary", width=6,
                    solidHeader=TRUE, DTOutput("xg_met")))
      ),
      
      tabItem(tabName = "xai",
              h2(class="section-title","💡 Explainable AI — Why did the model decide?"),
              fluidRow(
                box(title="🎯 Choose a car to explain", status="primary",
                    width=4, solidHeader=TRUE,
                    div(class="form-panel",
                        numericInput("xa_mpg", "MPG", 21, 5, 40, 0.1),
                        numericInput("xa_cyl", "Cylinders", 6, 4, 8),
                        numericInput("xa_hp",  "Horsepower", 110, 40, 400),
                        numericInput("xa_wt",  "Weight", 2.6, 1, 6, 0.05),
                        actionButton("xa_predict", "Explain",
                                     icon = icon("lightbulb"),
                                     class = "btn-success", width = "100%"))),
                box(title="Prediction", status="success", width=4,
                    solidHeader=TRUE, uiOutput("xa_result")),
                box(title="Global Feature Importance", status="info", width=4,
                    solidHeader=TRUE, plotOutput("xa_glob", height=300))),
              fluidRow(box(title="Local Contribution — Why this prediction?",
                           status="warning", width=12, solidHeader=TRUE,
                           plotOutput("xa_local", height=380)))
      ),
      
      tabItem(tabName = "proj",
              h2(class="section-title","🛒 Project — E-Commerce Transactions"),
              fluidRow(valueBoxOutput("pj_rev", width=3), valueBoxOutput("pj_txn", width=3),
                       valueBoxOutput("pj_avg", width=3), valueBoxOutput("pj_top", width=3)),
              fluidRow(
                box(title="Revenue by Category", status="primary", width=6,
                    solidHeader=TRUE, plotOutput("pj_cat")),
                box(title="Payment Method Frequency", status="success",
                    width=6, solidHeader=TRUE, plotOutput("pj_pay"))),
              fluidRow(
                box(title="Total Amount Distribution", status="warning",
                    width=6, solidHeader=TRUE, plotOutput("pj_hist")),
                box(title="Monthly Revenue Trend", status="info", width=6,
                    solidHeader=TRUE, plotOutput("pj_month"))),
              fluidRow(
                box(title="Spend by Membership", status="danger", width=6,
                    solidHeader=TRUE, plotOutput("pj_mem")),
                box(title="Price vs Total Amount", status="primary", width=6,
                    solidHeader=TRUE, plotOutput("pj_sp")),
                fluidRow(box(title="Revenue Summary", status="success",
                             width=12, solidHeader=TRUE, DTOutput("pj_tab"))))
      )
    )
  )
)

# =============================================================================
#  SERVER
# =============================================================================
server <- function(input, output, session) {
  
  DATA <- list(
    dtcln = gen_dtcln(), gss = gen_gss(), students = gen_students(),
    emp = gen_employees(), ts = gen_ts(), mall = gen_mall(),
    pca = gen_pca(), loan = gen_loan(), diab = gen_diabetes(),
    eco = gen_eco())
  
  # ---- Pre-fit models -----------------------------------------------------
  KNN <- tryCatch({
    set.seed(123); d <- DATA$loan
    idx <- createDataPartition(d$Loan_Approved, p = .8, list = FALSE)
    tr <- d[idx,]; te <- d[-idx,]
    feats <- c("Age","Income","Credit_Score","Loan_Amount")
    mins <- sapply(tr[feats], min); maxs <- sapply(tr[feats], max)
    mus  <- sapply(tr[feats], mean); sds <- sapply(tr[feats], sd); sds[sds==0] <- 1
    zs <- function(df) as.data.frame(mapply(function(col, mu, sd) (col - mu)/sd,
                                            df, mus, sds, SIMPLIFY = FALSE))
    trx <- zs(tr[feats]); tex <- zs(te[feats])
    ks <- 1:20
    acc <- sapply(ks, function(k)
      mean(class::knn(trx, tex, tr$Loan_Approved, k = k) == te$Loan_Approved))
    pred5 <- class::knn(trx, tex, tr$Loan_Approved, k = 5)
    list(ok = TRUE, feats = feats, mins = mins, maxs = maxs, mus = mus, sds = sds,
         trx = trx, tr_labels = tr$Loan_Approved, ks = ks, acc = acc,
         pred = pred5, actual = te$Loan_Approved)
  }, error = function(e) list(ok = FALSE))
  
  LOGIT <- tryCatch({
    set.seed(123); d <- DATA$diab
    idx <- createDataPartition(d$diabetes, p = .8, list = FALSE)
    tr <- d[idx,]; te <- d[-idx,]
    m <- glm(diabetes ~ age + hypertension + heart_disease + BMI +
               HbA1c_level + blood_glucose_level, data = tr, family = binomial)
    prob <- predict(m, te, type = "response")
    pred <- factor(ifelse(prob > .5, "Yes", "No"), levels = c("No","Yes"))
    list(ok = TRUE, m = m, pred = pred, prob = prob, actual = te$diabetes)
  }, error = function(e) list(ok = FALSE))
  
  SVM <- tryCatch({
    set.seed(123)
    idx <- createDataPartition(iris$Species, p = .8, list = FALSE)
    tr <- iris[idx,]; te <- iris[-idx,]
    m <- svm(Species ~ ., data = tr, kernel = "radial", probability = TRUE)
    list(ok = TRUE, m = m, pred = predict(m, te), actual = te$Species)
  }, error = function(e) list(ok = FALSE))
  
  RF_REG <- tryCatch({
    set.seed(123)
    idx <- createDataPartition(mtcars$mpg, p = .8, list = FALSE)
    tr <- mtcars[idx,]; te <- mtcars[-idx,]
    m <- randomForest(mpg ~ ., data = tr, ntree = 500, mtry = 3, importance = TRUE)
    list(ok = TRUE, m = m, pred = predict(m, te), actual = te$mpg)
  }, error = function(e) list(ok = FALSE))
  
  RF_CLS <- tryCatch({
    set.seed(123); idx <- sample(nrow(iris), .7*nrow(iris))
    m <- randomForest(Species ~ ., data = iris[idx,],
                      ntree = 500, mtry = 2, importance = TRUE)
    list(ok = TRUE, m = m, pred = predict(m, iris[-idx,]),
         actual = iris[-idx,]$Species)
  }, error = function(e) list(ok = FALSE))
  
  # ---------- FIXED XGBoost training ----------
  XGB <- tryCatch({
    set.seed(123); d <- DATA$diab
    idx <- createDataPartition(d$diabetes, p = .8, list = FALSE)
    tr <- d[idx,]; te <- d[-idx,]
    
    # Lock factor levels so dummyVars + predict are stable
    lvls <- list(
      hypertension     = c("No","Yes"),
      heart_disease    = c("No","Yes"),
      smoking_history  = c("never","former","current","not current","ever","No Info"))
    for (v in names(lvls)) {
      tr[[v]] <- factor(tr[[v]], levels = lvls[[v]])
      te[[v]] <- factor(te[[v]], levels = lvls[[v]])
    }
    
    # Encode predictors only (NOT the response)
    predictors <- setdiff(names(tr), "diabetes")
    dummy <- dummyVars("~ .", data = tr[, predictors], fullRank = FALSE)
    trx <- predict(dummy, tr[, predictors])
    tex <- predict(dummy, te[, predictors])
    
    ty <- as.integer(tr$diabetes) - 1L
    sy <- as.integer(te$diabetes) - 1L
    m <- xgb.train(list(objective = "binary:logistic", eval_metric = "logloss",
                        max_depth = 4, eta = .1, subsample = .8,
                        colsample_bytree = .8),
                   xgb.DMatrix(trx, label = ty), nrounds = 80, verbose = 0)
    prob <- predict(m, xgb.DMatrix(tex))
    pred <- factor(ifelse(prob > .5, "Yes", "No"), levels = c("No","Yes"))
    list(ok = TRUE, m = m, prob = prob, pred = pred, actual = te$diabetes,
         dummy = dummy, feats = colnames(trx), predictors = predictors)
  }, error = function(e) { message("XGB train failed: ", conditionMessage(e)); list(ok = FALSE) })
  
  # ---------- FIXED XAI training ----------
  # Train ONLY on the 4 features the UI exposes + am
  XAI <- tryCatch({
    d <- mtcars[, c("mpg","cyl","hp","wt","am")]
    d$am <- factor(d$am, levels = c(0,1), labels = c("Auto","Manual"))
    set.seed(42)
    m <- randomForest(am ~ ., data = d, ntree = 300, importance = TRUE)
    imp <- importance(m)
    imp_df <- data.frame(Variable = rownames(imp),
                         MeanDecreaseGini = imp[, "MeanDecreaseGini"])
    imp_df <- imp_df[order(-imp_df$MeanDecreaseGini), ]
    list(ok = TRUE, m = m, imp = imp_df, train_data = d,
         feats = c("mpg","cyl","hp","wt"))
  }, error = function(e) { message("XAI train failed: ", conditionMessage(e)); list(ok = FALSE) })
  
  KM <- tryCatch({
    set.seed(42); d <- scale(DATA$mall[, 2:5])
    list(ok = TRUE, scaled = d, model = kmeans(d, centers = 3, nstart = 25))
  }, error = function(e) list(ok = FALSE))
  
  PCA <- tryCatch(
    list(ok = TRUE, model = prcomp(DATA$pca, center = TRUE, scale. = TRUE)),
    error = function(e) list(ok = FALSE))
  
  TS <- tryCatch({
    obj <- ts(DATA$ts$Temperature, frequency = 12, start = c(2024, 1))
    list(ok = TRUE, obj = obj, fit = auto.arima(obj))
  }, error = function(e) list(ok = FALSE))
  
  # =========================================================
  #  CLEANING
  # =========================================================
  clean_res <- reactive({
    tryCatch({
      d <- DATA$dtcln; nb <- sum(is.na(d))
      cont <- c("Age","Income_KES","Household_Size","Farm_Size_Acres",
                "Monthly_Expenses_KES","Years_Experience")
      catv <- c("Education","Gender","Extension_Access","Adoption_Status")
      gm <- function(x){ux<-unique(na.omit(x));ux[which.max(tabulate(match(x,ux)))]}
      outc <- sapply(d[cont], function(x){
        q1<-quantile(x,.25,na.rm=TRUE);q3<-quantile(x,.75,na.rm=TRUE);iqr<-q3-q1
        sum(x < q1-1.5*iqr | x > q3+1.5*iqr, na.rm=TRUE)})
      for (v in cont){
        d[[v]][is.na(d[[v]])] <- median(d[[v]], na.rm = TRUE)
        q1 <- quantile(d[[v]], .25, na.rm=TRUE); q3 <- quantile(d[[v]], .75, na.rm=TRUE)
        iqr <- q3 - q1
        i <- which(d[[v]] < q1-1.5*iqr | d[[v]] > q3+1.5*iqr)
        d[[v]][i] <- median(d[[v]], na.rm = TRUE)
      }
      for (v in catv) d[[v]][is.na(d[[v]])] <- gm(d[[v]])
      list(before = DATA$dtcln, after = d, missing_before = nb,
           outliers = outc, cont = cont)
    }, error = function(e) list(before = DATA$dtcln, after = DATA$dtcln,
                                missing_before = 0,
                                outliers = setNames(rep(0,6),
                                                    c("Age","Income_KES","Household_Size",
                                                      "Farm_Size_Acres","Monthly_Expenses_KES",
                                                      "Years_Experience")),
                                cont = c("Age","Income_KES","Household_Size",
                                         "Farm_Size_Acres","Monthly_Expenses_KES",
                                         "Years_Experience")))
  })
  
  output$cl_miss <- renderValueBox(valueBox(clean_res()$missing_before,
                                            "Missing before", icon = icon("circle-question"), color = "red"))
  output$cl_outl <- renderValueBox(valueBox(sum(clean_res()$outliers),
                                            "Outliers replaced", icon = icon("triangle-exclamation"), color = "yellow"))
  output$cl_rows <- renderValueBox(valueBox(nrow(clean_res()$after),
                                            "Rows", icon = icon("table"), color = "green"))
  output$cl_cols <- renderValueBox(valueBox(ncol(clean_res()$after),
                                            "Columns", icon = icon("columns"), color = "purple"))
  
  output$cl_na <- renderPlot({
    tryCatch({
      df <- data.frame(Column = names(clean_res()$before),
                       Missing = colSums(is.na(clean_res()$before)))
      df <- df[df$Missing > 0, , drop = FALSE]
      if (!nrow(df)) {
        plot.new(); text(.5,.5,"✓ No missing values", cex=1.8, col="#27ae60")
        return()
      }
      print(ggplot(df, aes(reorder(Column, Missing), Missing, fill = Column)) +
              geom_col(show.legend = FALSE) + coord_flip() +
              theme_minimal(base_size = 13) +
              labs(x = NULL, y = "Count", title = "Missing values by column"))
    }, error = function(e) {
      plot.new(); text(.5,.5, paste("⚠", conditionMessage(e)), cex=1.2, col="grey40")
    })
  })
  
  output$cl_box <- renderPlot({
    tryCatch({
      d <- clean_res()$after[clean_res()$cont]
      long <- tidyr::pivot_longer(d, everything(),
                                  names_to = "var", values_to = "value")
      print(ggplot(long, aes(var, value, fill = var)) +
              geom_boxplot(outlier.colour = "red", outlier.size = 1.4) +
              coord_flip() + theme_minimal(base_size = 13) +
              theme(legend.position = "none") +
              labs(x = NULL, y = NULL, title = "Distribution after cleaning"))
    }, error = function(e) {
      plot.new(); text(.5,.5, paste("⚠", conditionMessage(e)), cex=1.2, col="grey40")
    })
  })
  
  output$cl_outliers <- renderDT({
    tryCatch({
      datatable(data.frame(Variable = names(clean_res()$outliers),
                           Outliers_Replaced = as.integer(clean_res()$outliers)),
                options = list(dom = "t"), rownames = FALSE)
    }, error = function(e)
      datatable(data.frame(Message = paste("⚠", conditionMessage(e))),
                options = list(dom = "t")))
  })
  
  output$cl_preview <- renderDT({
    tryCatch({
      datatable(head(clean_res()$after, 15),
                options = list(scrollX = TRUE, pageLength = 8), rownames = FALSE)
    }, error = function(e)
      datatable(data.frame(Message = paste("⚠", conditionMessage(e)))))
  })
  
  # =========================================================
  #  WRANGLING
  # =========================================================
  output$wr_filt <- renderDT({
    tryCatch({
      df <- DATA$gss %>% filter(marital %in% c("MARRIED","DIVORCED"),
                                degree %in% c("BACHELOR","GRADUATE","JUNIOR COLLEGE"),
                                sex == "MALE")
      datatable(head(df, 20), options = list(scrollX = TRUE, pageLength = 8),
                rownames = FALSE)
    }, error = function(e)
      datatable(data.frame(Message = paste("⚠", conditionMessage(e)))))
  })
  
  jt <- list(
    df1 = data.frame(id = 1:4, name = c("Alice","Bob","Charlie","David")),
    df2 = data.frame(id = c(2,3,5), score = c(85, 90, 78)))
  output$j_inner <- renderDT(datatable(inner_join(jt$df1, jt$df2, by = "id"), options=list(dom="t")))
  output$j_left  <- renderDT(datatable(left_join (jt$df1, jt$df2, by = "id"), options=list(dom="t")))
  output$j_right <- renderDT(datatable(right_join(jt$df1, jt$df2, by = "id"), options=list(dom="t")))
  output$j_full  <- renderDT(datatable(full_join (jt$df1, jt$df2, by = "id"), options=list(dom="t")))
  output$j_semi  <- renderDT(datatable(semi_join (jt$df1, jt$df2, by = "id"), options=list(dom="t")))
  output$j_anti  <- renderDT(datatable(anti_join (jt$df1, jt$df2, by = "id"), options=list(dom="t")))
  
  gw <- tibble::tibble(student = c("Alice","Bob","Charlie"),
                       math_2023 = c(85,92,NA), math_2024 = c(88,90,85),
                       science_2023 = c(78,NA,80), science_2024 = c(82,85,88))
  gl <- tidyr::pivot_longer(gw, -student,
                            names_to = c("subject","year"), names_sep = "_", values_to = "score")
  output$pv_long <- renderDT(datatable(gl, options=list(dom="t")))
  output$pv_wide <- renderDT(datatable(
    tidyr::pivot_wider(gl, names_from = c(subject, year),
                       names_sep = "_", values_from = score), options=list(dom="t")))
  
  # =========================================================
  #  VIZ
  # =========================================================
  output$vz_plot <- renderPlot({
    tryCatch({
      d <- DATA$students
      p <- switch(input$vz_type %||% "Histogram",
                  "Histogram" = ggplot(d, aes(GPA)) +
                    geom_histogram(bins = 25, fill = "steelblue", color = "white") +
                    geom_vline(xintercept = mean(d$GPA), color = "red", linetype = "dashed") +
                    theme_minimal(base_size = 13) + labs(title = "GPA distribution"),
                  "Boxplot" = ggplot(d, aes(Gender, Study_Hours, fill = Gender)) +
                    geom_boxplot(alpha = .75, outlier.color = "red") +
                    theme_minimal(base_size = 13) + theme(legend.position = "none") +
                    labs(title = "Study hours by gender"),
                  "Scatter+Fit" = ggplot(d, aes(Study_Hours, GPA)) +
                    geom_point(alpha = .5, color = "steelblue", size = 2.4) +
                    geom_smooth(method = "lm", color = "red", se = TRUE) +
                    theme_minimal(base_size = 13) +
                    labs(title = paste0("Study Hours vs GPA (r = ",
                                        round(cor(d$Study_Hours, d$GPA), 3), ")")),
                  "Violin" = ggplot(d, aes(Major, GPA, fill = Major)) +
                    geom_violin(trim = FALSE, alpha = .7) +
                    geom_boxplot(width = .15, fill = "white") +
                    theme_minimal(base_size = 13) + theme(legend.position = "none") +
                    labs(title = "GPA by major"),
                  "Ridge" = ggplot(d, aes(GPA, Major, fill = Major)) +
                    ggridges::geom_density_ridges(alpha = .7) +
                    theme_minimal(base_size = 13) + theme(legend.position = "none") +
                    labs(title = "GPA distribution by major"),
                  "Correlation Heatmap" = {
                    m <- cor(DATA$gss[, c("income","age","height","weight","hrswrk")])
                    melted <- reshape2::melt(round(m, 2))
                    ggplot(melted, aes(Var1, Var2, fill = value)) +
                      geom_tile(color = "white") + geom_text(aes(label = value), size = 4) +
                      scale_fill_gradient2(low="tomato", mid="white", high="steelblue",
                                           midpoint = 0, limits = c(-1,1)) +
                      coord_fixed() + theme_minimal(base_size = 12) +
                      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
                      labs(x = NULL, y = NULL, title = "Correlation heatmap")},
                  "Pairs Plot" = ggpairs(DATA$students[, c("Study_Hours","GPA",
                                                           "Pre_Test_Score","Post_Test_Score")]),
                  "Pie" = {
                    d2 <- DATA$gss %>% count(sex) %>% mutate(pct = round(100*n/sum(n),1),
                                                             lbl = paste0(pct,"%"))
                    ggplot(d2, aes("", n, fill = sex)) + geom_col(width = 1) +
                      coord_polar("y") + geom_text(aes(label = lbl),
                                                   position = position_stack(vjust = .5), size = 6, color = "white") +
                      theme_void(base_size = 13) + labs(title = "Gender split")})
      if (inherits(p, "ggplot")) print(p) else print(p)
    }, error = function(e) {
      plot.new(); text(.5,.5, paste("⚠", conditionMessage(e)), cex=1.2, col="grey40")
    })
  })
  
  output$vz_summary <- renderDT({
    tryCatch({
      d <- DATA$students %>% group_by(Major) %>%
        summarise(N = n(), Mean_GPA = round(mean(GPA), 3),
                  Median_GPA = round(median(GPA), 3),
                  Mean_Hours = round(mean(Study_Hours), 2))
      datatable(d, options = list(dom = "t"), rownames = FALSE)
    }, error = function(e)
      datatable(data.frame(Message = paste("⚠", conditionMessage(e)))))
  })
  
  # =========================================================
  #  STATISTICS
  # =========================================================
  output$st_out <- renderPrint({
    tryCatch({
      m1 <- lm(Productivity_Score ~ Training_Hours, data = DATA$emp)
      cat("=== Simple: Productivity ~ Training ===\n"); print(coef(summary(m1)))
      cat("\n=== Multiple: + Experience + Stress ===\n")
      m <- lm(Productivity_Score ~ Years_Experience + Training_Hours + Stress_Index,
              data = DATA$emp)
      print(coef(summary(m)))
      cat("\nR² =", round(summary(m)$r.squared, 4),
          "  Adj R² =", round(summary(m)$adj.r.squared, 4), "\n")
      cat("\n=== T-test: Productivity vs 60 ===\n")
      print(t.test(DATA$emp$Productivity_Score, mu = 60))
    }, error = function(e) cat("Error:", conditionMessage(e), "\n"))
  })
  
  output$st_coef <- renderDT({
    tryCatch({
      m <- lm(Productivity_Score ~ Years_Experience + Training_Hours + Stress_Index,
              data = DATA$emp)
      cf <- coef(summary(m))
      df <- data.frame(Term = rownames(cf), cf, row.names = NULL)
      names(df) <- c("Term","Estimate","Std.Error","t.value","p.value")
      df[, -1] <- round(df[, -1], 4)
      datatable(df, options = list(dom = "t"), rownames = FALSE)
    }, error = function(e)
      datatable(data.frame(Message = paste("⚠", conditionMessage(e)))))
  })
  
  output$st_plot <- renderPlot({
    tryCatch({
      print(ggplot(DATA$emp, aes(Training_Hours, Productivity_Score)) +
              geom_point(alpha = .5, color = "steelblue") +
              geom_smooth(method = "lm", color = "red") +
              theme_minimal(base_size = 13) + labs(title = "Training vs Productivity"))
    }, error = function(e) {
      plot.new(); text(.5,.5, paste("⚠", conditionMessage(e)), cex=1.2, col="grey40")
    })
  })
  
  output$st_diag <- renderPlot({
    tryCatch({
      m <- lm(Productivity_Score ~ Years_Experience + Training_Hours + Stress_Index,
              data = DATA$emp)
      par(mfrow = c(1,2))
      plot(m, which = 1, main = "Residuals vs Fitted")
      plot(m, which = 2, main = "Q-Q Plot")
      par(mfrow = c(1,1))
    }, error = function(e) {
      plot.new(); text(.5,.5, paste("⚠", conditionMessage(e)), cex=1.2, col="grey40")
    })
  })
  
  output$st_anova <- renderPrint({
    tryCatch({
      print(summary(aov(Productivity_Score ~ Department, data = DATA$emp)))
      cat("\nChi-square: Major vs Passed_Class\n")
      print(chisq.test(table(DATA$students$Major, DATA$students$Passed_Class)))
    }, error = function(e) cat("Error:", conditionMessage(e), "\n"))
  })
  
  # =========================================================
  #  TIME SERIES
  # =========================================================
  ts_forecast <- reactive({
    if (!isTRUE(TS$ok)) return(NULL)
    tryCatch(forecast(TS$fit, h = input$ts_h %||% 12), error = function(e) NULL)
  })
  
  output$ts_plot <- renderPlot({
    tryCatch({
      f <- ts_forecast()
      if (is.null(f)) { plot.new(); text(.5,.5,"Not available"); return() }
      print(autoplot(f) + theme_minimal(base_size = 13) +
              labs(title = "Temperature + ARIMA forecast", y = "Temp (°C)"))
    }, error = function(e) {
      plot.new(); text(.5,.5, paste("⚠", conditionMessage(e)), cex=1.2, col="grey40")
    })
  })
  
  output$ts_dec <- renderPlot({
    tryCatch({
      if (!isTRUE(TS$ok)) { plot.new(); text(.5,.5,"Not available"); return() }
      plot(decompose(TS$obj))
    }, error = function(e) {
      plot.new(); text(.5,.5, paste("⚠", conditionMessage(e)), cex=1.2, col="grey40")
    })
  })
  
  output$ts_acf <- renderPlot({
    tryCatch({
      if (!isTRUE(TS$ok)) { plot.new(); text(.5,.5,"Not available"); return() }
      par(mfrow = c(2,1)); acf(TS$obj, 30, main="ACF")
      pacf(TS$obj, 30, main="PACF"); par(mfrow = c(1,1))
    }, error = function(e) {
      plot.new(); text(.5,.5, paste("⚠", conditionMessage(e)), cex=1.2, col="grey40")
    })
  })
  
  output$ts_adf <- renderPrint({
    tryCatch({
      if (!isTRUE(TS$ok)) { cat("Not available\n"); return() }
      cat("Raw series:\n"); print(adf.test(TS$obj))
      cat("\n1st difference:\n"); print(adf.test(diff(TS$obj)))
    }, error = function(e) cat("Error:", conditionMessage(e), "\n"))
  })
  
  output$ts_forecast_tbl <- renderDT({
    tryCatch({
      f <- ts_forecast()
      if (is.null(f)) return(datatable(data.frame(Message = "Not available")))
      df <- data.frame(Period = as.character(time(f$mean)),
                       Forecast = round(as.numeric(f$mean), 2),
                       Lo80 = round(as.numeric(f$lower[,1]), 2),
                       Hi80 = round(as.numeric(f$upper[,1]), 2),
                       Lo95 = round(as.numeric(f$lower[,2]), 2),
                       Hi95 = round(as.numeric(f$upper[,2]), 2))
      datatable(df, options = list(dom = "tp", pageLength = 12), rownames = FALSE)
    }, error = function(e)
      datatable(data.frame(Message = paste("⚠", conditionMessage(e)))))
  })
  
  # =========================================================
  #  UNSUPERVISED
  # =========================================================
  output$km_elbow <- renderPlot({
    tryCatch({
      if (!isTRUE(KM$ok)) { plot.new(); text(.5,.5,"Not available"); return() }
      print(fviz_nbclust(KM$scaled, kmeans, method = "wss") +
              labs(title = "Elbow method") + theme_minimal(base_size = 13))
    }, error = function(e) {
      plot.new(); text(.5,.5, paste("⚠", conditionMessage(e)), cex=1.2, col="grey40")
    })
  })
  
  output$km_plot <- renderPlot({
    tryCatch({
      if (!isTRUE(KM$ok)) { plot.new(); text(.5,.5,"Not available"); return() }
      print(fviz_cluster(KM$model, data = KM$scaled, geom = "point",
                         ellipse.type = "convex",
                         ggtheme = theme_minimal(base_size = 12)) +
              labs(title = "K-Means clusters"))
    }, error = function(e) {
      plot.new(); text(.5,.5, paste("⚠", conditionMessage(e)), cex=1.2, col="grey40")
    })
  })
  
  output$km_summary <- renderDT({
    tryCatch({
      if (!isTRUE(KM$ok)) return(datatable(data.frame(Message = "Not available")))
      d <- DATA$mall; d$Cluster <- factor(KM$model$cluster)
      df <- d %>% group_by(Cluster) %>%
        summarise(N = n(), Mean_Age = round(mean(Age), 1),
                  Mean_Income = round(mean(Annual_Income), 0),
                  Mean_Spend = round(mean(Spending_Score), 1),
                  Mean_Purchase = round(mean(Average_Purchase_Value), 1))
      datatable(df, options = list(dom = "t"), rownames = FALSE)
    }, error = function(e)
      datatable(data.frame(Message = paste("⚠", conditionMessage(e)))))
  })
  
  output$pc_scree <- renderPlot({
    tryCatch({
      if (!isTRUE(PCA$ok)) { plot.new(); text(.5,.5,"Not available"); return() }
      eig <- PCA$model$sdev^2
      df <- data.frame(PC = factor(paste0("PC", seq_along(eig)),
                                   levels = paste0("PC", seq_along(eig))),
                       Eigen = eig)
      print(ggplot(df, aes(PC, Eigen)) + geom_col(fill = "steelblue") +
              geom_point(color = "red") + geom_line(group = 1, color = "red",
                                                    linetype = "dashed") +
              geom_hline(yintercept = 1, linetype = "dotted", color = "darkgreen") +
              theme_minimal(base_size = 13) + labs(title = "Scree plot", y = "Eigenvalue"))
    }, error = function(e) {
      plot.new(); text(.5,.5, paste("⚠", conditionMessage(e)), cex=1.2, col="grey40")
    })
  })
  
  output$pc_biplot <- renderPlot({
    tryCatch({
      if (!isTRUE(PCA$ok)) { plot.new(); text(.5,.5,"Not available"); return() }
      print(fviz_pca_biplot(PCA$model, geom = "point",
                            ggtheme = theme_minimal(base_size = 12)) +
              ggtitle("PCA Biplot"))
    }, error = function(e) {
      plot.new(); text(.5,.5, paste("⚠", conditionMessage(e)), cex=1.2, col="grey40")
    })
  })
  
  output$pc_var <- renderDT({
    tryCatch({
      if (!isTRUE(PCA$ok)) return(datatable(data.frame(Message = "Not available")))
      eig <- PCA$model$sdev^2; v <- eig/sum(eig)*100; c <- cumsum(v)
      datatable(data.frame(Component = paste0("PC", seq_along(eig)),
                           Eigenvalue = round(eig, 3),
                           Var_pct = round(v, 2),
                           Cum_pct = round(c, 2)),
                options = list(dom = "t"), rownames = FALSE)
    }, error = function(e)
      datatable(data.frame(Message = paste("⚠", conditionMessage(e)))))
  })
  
  output$pc_red <- renderDT({
    tryCatch({
      if (!isTRUE(PCA$ok)) return(datatable(data.frame(Message = "Not available")))
      datatable(round(as.data.frame(PCA$model$x[, 1:4]), 3),
                options = list(pageLength = 8, scrollX = TRUE), rownames = FALSE)
    }, error = function(e)
      datatable(data.frame(Message = paste("⚠", conditionMessage(e)))))
  })
  
  # =========================================================
  #  KNN
  # =========================================================
  output$kn_acc <- renderPlot({
    tryCatch({
      if (!isTRUE(KNN$ok)) { plot.new(); text(.5,.5,"Not available"); return() }
      print(ggplot(data.frame(k = KNN$ks, Accuracy = KNN$acc), aes(k, Accuracy)) +
              geom_line(color = "darkgreen") + geom_point(color = "red") +
              theme_minimal(base_size = 13) + labs(title = "KNN accuracy vs k"))
    }, error = function(e) {
      plot.new(); text(.5,.5, paste("⚠", conditionMessage(e)), cex=1.2, col="grey40")
    })
  })
  
  output$kn_cm <- renderPlot({
    tryCatch({
      if (!isTRUE(KNN$ok)) { plot.new(); text(.5,.5,"Not available"); return() }
      cm <- caret::confusionMatrix(KNN$pred, KNN$actual)
      fourfoldplot(cm$table, color = c("#F8766D","#00BFC4"),
                   main = paste0("k = 5 (Accuracy = ",
                                 round(cm$overall["Accuracy"], 4), ")"))
    }, error = function(e) {
      plot.new(); text(.5,.5, paste("⚠", conditionMessage(e)), cex=1.2, col="grey40")
    })
  })
  
  output$kn_pred <- renderDT({
    tryCatch({
      if (!isTRUE(KNN$ok)) return(datatable(data.frame(Message = "Not available")))
      df <- data.frame(Actual = KNN$actual, Predicted = KNN$pred,
                       Correct = KNN$actual == KNN$pred)
      datatable(head(df, 30), options = list(pageLength = 8, scrollX = TRUE),
                rownames = FALSE) %>%
        formatStyle("Correct", backgroundColor = styleEqual(c(TRUE,FALSE),
                                                            c("#d4edda","#f8d7da")))
    }, error = function(e)
      datatable(data.frame(Message = paste("⚠", conditionMessage(e)))))
  })
  
  observeEvent(input$kn_predict, {
    output$kn_result <- renderUI({
      tryCatch({
        if (!isTRUE(KNN$ok))
          return(div(class = "pred-box warn", "KNN model not available"))
        newx <- data.frame(Age = input$kn_age, Income = input$kn_inc,
                           Credit_Score = input$kn_cs, Loan_Amount = input$kn_loan)
        newx_clip <- as.data.frame(mapply(function(col, mn, mx)
          pmin(pmax(col, mn), mx), newx, KNN$mins, KNN$maxs, SIMPLIFY = FALSE))
        newx_z <- as.data.frame(mapply(function(col, mu, sd)
          (col - mu)/sd, newx_clip, KNN$mus, KNN$sds, SIMPLIFY = FALSE))
        k_use <- 7
        d <- as.matrix(stats::dist(rbind(KNN$trx, newx_z)))[
          nrow(KNN$trx) + 1, seq_len(nrow(KNN$trx))]
        idx <- order(d)[seq_len(k_use)]
        w <- 1/(d[idx] + 1e-6)
        labs <- as.character(KNN$tr_labels[idx])
        w_yes <- sum(w[labs == "Yes"]); w_no <- sum(w[labs == "No"])
        prob_yes <- w_yes/(w_yes + w_no)
        cls <- ifelse(prob_yes >= 0.5, "Yes", "No")
        conf <- max(prob_yes, 1 - prob_yes)
        hint <- if (newx$Income[1] > 43000) "Yes (income rule)" else "No (income rule)"
        div(class = paste("pred-box", if (cls == "Yes") "" else "info"),
            div(class = "lbl", "KNN says"),
            div(class = "val", paste0(cls, " Loan")),
            div(class = "lbl", paste0("Confidence: ", round(conf*100, 1), "%")),
            tags$hr(style="border-top:1px solid rgba(255,255,255,.3);margin:10px 0;"),
            div(style = "font-size:12px;",
                tags$b("Income rule: "), hint))
      }, error = function(e)
        div(class = "pred-box warn", "Error: ", conditionMessage(e)))
    })
  })
  
  # =========================================================
  #  LOGISTIC
  # =========================================================
  output$lg_coef <- renderDT({
    tryCatch({
      if (!isTRUE(LOGIT$ok)) return(datatable(data.frame(Message = "Not available")))
      cf <- coef(summary(LOGIT$m))
      df <- data.frame(Term = rownames(cf), cf, row.names = NULL)
      names(df) <- c("Term","Estimate","Std.Error","z.value","p.value")
      df[, -1] <- round(df[, -1], 4)
      datatable(df, options = list(pageLength = 8, dom = "tp"), rownames = FALSE)
    }, error = function(e)
      datatable(data.frame(Message = paste("⚠", conditionMessage(e)))))
  })
  
  output$lg_cm <- renderPlot({
    tryCatch({
      if (!isTRUE(LOGIT$ok)) { plot.new(); text(.5,.5,"Not available"); return() }
      cm <- caret::confusionMatrix(LOGIT$pred, LOGIT$actual, positive = "Yes")
      fourfoldplot(cm$table, color = c("#F8766D","#00BFC4"),
                   main = paste0("Accuracy = ", round(cm$overall["Accuracy"], 4)))
    }, error = function(e) {
      plot.new(); text(.5,.5, paste("⚠", conditionMessage(e)), cex=1.2, col="grey40")
    })
  })
  
  output$lg_roc <- renderPlot({
    tryCatch({
      if (!isTRUE(LOGIT$ok)) { plot.new(); text(.5,.5,"Not available"); return() }
      r <- pROC::roc(LOGIT$actual, LOGIT$prob, levels = c("No","Yes"), quiet = TRUE)
      plot(r, col = "steelblue", lwd = 2,
           main = paste0("AUC = ", round(pROC::auc(r), 4)))
    }, error = function(e) {
      plot.new(); text(.5,.5, paste("⚠", conditionMessage(e)), cex=1.2, col="grey40")
    })
  })
  
  observeEvent(input$lg_predict, {
    output$lg_result <- renderUI({
      tryCatch({
        if (!isTRUE(LOGIT$ok))
          return(div(class = "pred-box warn", "Model not available"))
        newd <- data.frame(age = input$lg_age,
                           hypertension = factor(input$lg_htn, levels = c("No","Yes")),
                           heart_disease = factor(input$lg_hd, levels = c("No","Yes")),
                           BMI = input$lg_bmi,
                           HbA1c_level = input$lg_hba,
                           blood_glucose_level = input$lg_glu)
        prob <- predict(LOGIT$m, newdata = newd, type = "response")
        cls <- ifelse(prob > .5, "Yes", "No")
        div(class = paste("pred-box", if (cls == "Yes") "" else "info"),
            div(class = "lbl", "Logistic says"),
            div(class = "val", paste0(cls, " Diabetes")),
            div(class = "lbl", paste0("P(Yes) = ", round(prob*100, 1), "%")))
      }, error = function(e)
        div(class = "pred-box warn", "Error: ", conditionMessage(e)))
    })
  })
  
  # =========================================================
  #  SVM
  # =========================================================
  output$sv_cm <- renderPlot({
    tryCatch({
      if (!isTRUE(SVM$ok)) { plot.new(); text(.5,.5,"Not available"); return() }
      cm <- caret::confusionMatrix(SVM$pred, SVM$actual)
      fourfoldplot(cm$table, color = c("#F8766D","#00BFC4"),
                   main = paste0("Accuracy = ", round(cm$overall["Accuracy"], 4)))
    }, error = function(e) {
      plot.new(); text(.5,.5, paste("⚠", conditionMessage(e)), cex=1.2, col="grey40")
    })
  })
  
  observeEvent(input$sv_predict, {
    output$sv_result <- renderUI({
      tryCatch({
        if (!isTRUE(SVM$ok))
          return(div(class = "pred-box warn", "SVM not available"))
        newd <- data.frame(Sepal.Length = input$sv_sl, Sepal.Width = input$sv_sw,
                           Petal.Length = input$sv_pl, Petal.Width = input$sv_pw)
        pred <- predict(SVM$m, newdata = newd, probability = TRUE)
        prob <- attr(pred, "probabilities")[1, ]
        div(class = "pred-box",
            div(class = "lbl", "SVM says"),
            div(class = "val", as.character(pred)),
            div(class = "lbl", paste0("Confidence: ", round(max(prob)*100, 1), "%")))
      }, error = function(e)
        div(class = "pred-box warn", "Error: ", conditionMessage(e)))
    })
  })
  
  # =========================================================
  #  RANDOM FOREST
  # =========================================================
  output$rf_imp <- renderPlot({
    tryCatch({
      if (!isTRUE(RF_REG$ok)) { plot.new(); text(.5,.5,"Not available"); return() }
      imp <- importance(RF_REG$m)
      imp_df <- data.frame(Variable = rownames(imp), IncMSE = imp[, "%IncMSE"])
      imp_df <- imp_df[order(-imp_df$IncMSE), ]
      print(ggplot(imp_df, aes(reorder(Variable, IncMSE), IncMSE, fill = Variable)) +
              geom_col(show.legend = FALSE) + coord_flip() +
              theme_minimal(base_size = 13) +
              labs(x = NULL, y = "%IncMSE", title = "RF importance (MPG)"))
    }, error = function(e) {
      plot.new(); text(.5,.5, paste("⚠", conditionMessage(e)), cex=1.2, col="grey40")
    })
  })
  
  output$rf_act <- renderPlot({
    tryCatch({
      if (!isTRUE(RF_REG$ok)) { plot.new(); text(.5,.5,"Not available"); return() }
      plot(RF_REG$actual, RF_REG$pred, pch = 19, col = "steelblue", cex = 1.4,
           xlab = "Actual MPG", ylab = "Predicted MPG",
           main = paste0("R² = ", round(cor(RF_REG$actual, RF_REG$pred)^2, 3)))
      abline(0, 1, col = "red", lwd = 2, lty = 2); grid()
    }, error = function(e) {
      plot.new(); text(.5,.5, paste("⚠", conditionMessage(e)), cex=1.2, col="grey40")
    })
  })
  
  output$rf_met <- renderDT({
    tryCatch({
      if (!isTRUE(RF_REG$ok)) return(datatable(data.frame(Message = "Not available")))
      df <- data.frame(Metric = c("RMSE","MAE","R²"),
                       Value = round(c(sqrt(mean((RF_REG$actual - RF_REG$pred)^2)),
                                       mean(abs(RF_REG$actual - RF_REG$pred)),
                                       cor(RF_REG$actual, RF_REG$pred)^2), 4))
      datatable(df, options = list(dom = "t"), rownames = FALSE)
    }, error = function(e)
      datatable(data.frame(Message = paste("⚠", conditionMessage(e)))))
  })
  
  observeEvent(input$rf_predict, {
    output$rf_result <- renderUI({
      tryCatch({
        if (!isTRUE(RF_REG$ok))
          return(div(class = "pred-box warn", "Model not available"))
        newd <- data.frame(cyl = input$rf_cyl, disp = input$rf_disp,
                           hp = input$rf_hp, drat = input$rf_drat,
                           wt = input$rf_wt, qsec = input$rf_qsec,
                           vs = input$rf_vs, am = input$rf_am,
                           gear = input$rf_gear, carb = input$rf_carb)
        pred <- predict(RF_REG$m, newdata = newd)
        div(class = "pred-box",
            div(class = "lbl", "Random Forest predicts"),
            div(class = "val", paste0(round(pred, 2), " MPG")),
            div(class = "lbl", "miles per gallon"))
      }, error = function(e)
        div(class = "pred-box warn", "Error: ", conditionMessage(e)))
    })
  })
  
  output$rf_iris <- renderPlot({
    tryCatch({
      if (!isTRUE(RF_CLS$ok)) { plot.new(); text(.5,.5,"Not available"); return() }
      cm <- caret::confusionMatrix(RF_CLS$pred, RF_CLS$actual)
      fourfoldplot(cm$table, color = c("#F8766D","#00BFC4"),
                   main = paste0("Accuracy = ", round(cm$overall["Accuracy"], 4)))
    }, error = function(e) {
      plot.new(); text(.5,.5, paste("⚠", conditionMessage(e)), cex=1.2, col="grey40")
    })
  })
  
  observeEvent(input$rfc_predict, {
    output$rfc_result <- renderUI({
      tryCatch({
        if (!isTRUE(RF_CLS$ok))
          return(div(class = "pred-box warn", "Model not available"))
        newd <- data.frame(Sepal.Length = input$rfc_sl, Sepal.Width = input$rfc_sw,
                           Petal.Length = input$rfc_pl, Petal.Width = input$rfc_pw)
        pred <- predict(RF_CLS$m, newdata = newd)
        prob <- predict(RF_CLS$m, newdata = newd, type = "prob")
        div(class = "pred-box",
            div(class = "lbl", "Random Forest predicts"),
            div(class = "val", as.character(pred)),
            div(class = "lbl", paste0("Confidence: ", round(max(prob)*100, 1), "%")))
      }, error = function(e)
        div(class = "pred-box warn", "Error: ", conditionMessage(e)))
    })
  })
  
  # =========================================================
  #  XGBOOST  — FIXED
  # =========================================================
  output$xg_roc <- renderPlot({
    tryCatch({
      if (!isTRUE(XGB$ok)) { plot.new(); text(.5,.5,"Not available"); return() }
      r <- pROC::roc(XGB$actual, XGB$prob, levels = c("No","Yes"), quiet = TRUE)
      plot(r, col = "darkgreen", lwd = 2,
           main = paste0("AUC = ", round(pROC::auc(r), 4)))
    }, error = function(e) {
      plot.new(); text(.5,.5, paste("⚠", conditionMessage(e)), cex=1.2, col="grey40")
    })
  })
  
  output$xg_imp <- renderPlot({
    tryCatch({
      if (!isTRUE(XGB$ok)) { plot.new(); text(.5,.5,"Not available"); return() }
      imp <- xgb.importance(XGB$feats, model = XGB$m)
      xgb.plot.importance(imp, main = "XGBoost importance", col = "steelblue")
    }, error = function(e) {
      plot.new(); text(.5,.5, paste("⚠", conditionMessage(e)), cex=1.2, col="grey40")
    })
  })
  
  output$xg_met <- renderDT({
    tryCatch({
      if (!isTRUE(XGB$ok)) return(datatable(data.frame(Message = "Not available")))
      cm <- caret::confusionMatrix(XGB$pred, XGB$actual, positive = "Yes")
      r <- pROC::roc(XGB$actual, XGB$prob, levels = c("No","Yes"), quiet = TRUE)
      df <- data.frame(Metric = c("Accuracy","Precision","Recall","F1",
                                  "Specificity","AUC"),
                       Value = round(c(cm$overall["Accuracy"],
                                       cm$byClass["Precision"],
                                       cm$byClass["Recall"],
                                       cm$byClass["F1"],
                                       cm$byClass["Specificity"],
                                       as.numeric(pROC::auc(r))), 4))
      datatable(df, options = list(dom = "t"), rownames = FALSE)
    }, error = function(e)
      datatable(data.frame(Message = paste("⚠", conditionMessage(e)))))
  })
  
  observeEvent(input$xg_predict, {
    output$xg_result <- renderUI({
      tryCatch({
        if (!isTRUE(XGB$ok))
          return(div(class = "pred-box warn", "Model not available"))
        
        newd <- data.frame(
          age = input$xg_age,
          hypertension = factor(input$xg_htn, levels = c("No","Yes")),
          heart_disease = factor(input$xg_hd, levels = c("No","Yes")),
          smoking_history = factor(input$xg_smoke,
                                   levels = c("never","former","current",
                                              "not current","ever","No Info")),
          BMI = input$xg_bmi,
          HbA1c_level = input$xg_hba,
          blood_glucose_level = input$xg_glu)
        
        # Reorder to exactly the training predictor columns
        newd <- newd[, XGB$predictors, drop = FALSE]
        
        newx <- predict(XGB$dummy, newd)
        # Force the same column order as training
        newx <- as.matrix(newx)[, XGB$feats, drop = FALSE]
        
        prob <- predict(XGB$m, xgb.DMatrix(newx))
        cls <- ifelse(prob > .5, "Yes", "No")
        
        div(class = paste("pred-box", if (cls == "Yes") "" else "info"),
            div(class = "lbl", "XGBoost predicts"),
            div(class = "val", paste0(cls, " Diabetes")),
            div(class = "lbl", paste0("P(Yes) = ", round(prob*100, 1), "%")))
      }, error = function(e)
        div(class = "pred-box warn", "Error: ", conditionMessage(e)))
    })
  })
  
  # =========================================================
  #  XAI 
  # =========================================================
  output$xa_glob <- renderPlot({
    tryCatch({
      if (!isTRUE(XAI$ok)) { plot.new(); text(.5,.5,"Not available"); return() }
      print(ggplot(XAI$imp, aes(reorder(Variable, MeanDecreaseGini),
                                MeanDecreaseGini, fill = Variable)) +
              geom_col(show.legend = FALSE) + coord_flip() +
              theme_minimal(base_size = 13) +
              labs(x = NULL, y = "Mean Decrease Gini", title = "Global importance"))
    }, error = function(e) {
      plot.new(); text(.5,.5, paste("⚠", conditionMessage(e)), cex=1.2, col="grey40")
    })
  })
  
  # Single, robust local-explanation reactive
  xai_explain <- eventReactive(input$xa_predict, {
    tryCatch({
      if (!isTRUE(XAI$ok)) return(NULL)
      
      newd <- data.frame(
        mpg = as.numeric(input$xa_mpg),
        cyl = as.numeric(input$xa_cyl),
        hp  = as.numeric(input$xa_hp),
        wt  = as.numeric(input$xa_wt))
      # Ensure the columns match training order exactly
      newd <- newd[, XAI$feats, drop = FALSE]
      
      pred_prob  <- predict(XAI$m, newd, type = "prob")
      pred_class <- colnames(pred_prob)[which.max(pred_prob[1, ])]
      base_p     <- as.numeric(pred_prob[1, pred_class])
      
      # Local contribution via mean-substitution (one-at-a-time perturbation)
      contribs <- sapply(XAI$feats, function(v) {
        pert <- newd
        pert[[v]] <- mean(XAI$train_data[[v]], na.rm = TRUE)
        p_pert <- as.numeric(predict(XAI$m, pert, type = "prob")[1, pred_class])
        base_p - p_pert                # >0 → variable pushes towards pred_class
      })
      
      list(ok = TRUE, pred_class = pred_class,
           pred_prob = pred_prob, base_p = base_p,
           contributions = contribs, vars = XAI$feats)
    }, error = function(e) {
      message("xai_explain FAILED: ", conditionMessage(e))
      NULL
    })
  })
  
  output$xa_result <- renderUI({
    tryCatch({
      r <- xai_explain()
      if (is.null(r) || !isTRUE(r$ok))
        return(div(class = "pred-box info",
                   div(class = "lbl", "Click Explain"),
                   div(class = "val", "—"),
                   div(class = "lbl", "to see prediction")))
      div(class = "pred-box",
          div(class = "lbl", "Random Forest predicts"),
          div(class = "val", r$pred_class),
          div(class = "lbl", paste0("Confidence: ",
                                    round(r$base_p*100, 1), "%")))
    }, error = function(e)
      div(class = "pred-box warn", "Error: ", conditionMessage(e)))
  })
  
  output$xa_local <- renderPlot({
    tryCatch({
      r <- xai_explain()
      if (is.null(r) || !isTRUE(r$ok)) {
        plot.new(); text(.5,.5,"Click Explain to see local contributions",
                         cex = 1.4, col = "grey40")
        return()
      }
      df <- data.frame(Variable = r$vars,
                       Contribution = as.numeric(r$contributions))
      other_class <- setdiff(levels(XAI$train_data$am), r$pred_class)
      df$Direction <- ifelse(
        df$Contribution > 0,
        paste0("Pushes towards ", r$pred_class),
        paste0("Pushes towards ", other_class))
      print(ggplot(df, aes(reorder(Variable, Contribution), Contribution,
                           fill = Direction)) +
              geom_col() + coord_flip() +
              scale_fill_manual(values = setNames(
                c("#27ae60","#e74c3c"),
                c(paste0("Pushes towards ", r$pred_class),
                  paste0("Pushes towards ", other_class)))) +
              geom_hline(yintercept = 0, color = "grey40") +
              theme_minimal(base_size = 13) +
              labs(x = NULL, y = "Δ Probability of predicted class",
                   title = paste0("Why did the model predict '",
                                  r$pred_class, "'?"),
                   subtitle = "Change in P(predicted) when feature is set to its training mean"))
    }, error = function(e) {
      plot.new(); text(.5,.5, paste("⚠", conditionMessage(e)), cex=1.2, col="grey40")
    })
  })
  
  # =========================================================
  #  PROJECT
  # =========================================================
  eco_r <- reactive({
    d <- DATA$eco
    d$Age[is.na(d$Age)] <- mean(d$Age, na.rm = TRUE)
    d
  })
  
  output$pj_rev <- renderValueBox(valueBox(
    paste0("$", format(round(sum(eco_r()$TotalAmount)), big.mark = ",")),
    "Total revenue", icon = icon("dollar-sign"), color = "green"))
  output$pj_txn <- renderValueBox(valueBox(nrow(eco_r()), "Transactions",
                                           icon = icon("receipt"), color = "purple"))
  output$pj_avg <- renderValueBox(valueBox(
    paste0("$", round(mean(eco_r()$TotalAmount), 0)),
    "Avg basket", icon = icon("shopping-cart"), color = "blue"))
  
  pj_top <- reactive({
    eco_r() %>% group_by(ProductCategory) %>%
      summarise(Rev = sum(TotalAmount)) %>% arrange(-Rev)
  })
  output$pj_top <- renderValueBox(valueBox(pj_top()$ProductCategory[1],
                                           "Top category", icon = icon("trophy"), color = "yellow"))
  
  output$pj_cat <- renderPlot({
    tryCatch({
      print(ggplot(pj_top(), aes(reorder(ProductCategory, Rev), Rev,
                                 fill = ProductCategory)) +
              geom_col() + coord_flip() + theme_minimal(base_size = 13) +
              theme(legend.position = "none") +
              scale_y_continuous(labels = scales::dollar) +
              labs(x = NULL, y = "Revenue", title = "Revenue by category"))
    }, error = function(e) {
      plot.new(); text(.5,.5, paste("⚠", conditionMessage(e)), cex=1.2, col="grey40")
    })
  })
  
  output$pj_pay <- renderPlot({
    tryCatch({
      df <- as.data.frame(table(eco_r()$PaymentMethod))
      names(df) <- c("Method","Count")
      print(ggplot(df, aes(reorder(Method, -Count), Count, fill = Method)) +
              geom_col() + coord_flip() + theme_minimal(base_size = 13) +
              theme(legend.position = "none") +
              labs(x = NULL, y = "Transactions", title = "Payment methods"))
    }, error = function(e) {
      plot.new(); text(.5,.5, paste("⚠", conditionMessage(e)), cex=1.2, col="grey40")
    })
  })
  
  output$pj_hist <- renderPlot({
    tryCatch({
      print(ggplot(eco_r(), aes(TotalAmount)) +
              geom_histogram(bins = 30, fill = "darkorange", color = "white") +
              geom_vline(xintercept = mean(eco_r()$TotalAmount),
                         color = "red", linetype = "dashed") +
              geom_vline(xintercept = median(eco_r()$TotalAmount),
                         color = "blue", linetype = "dotted") +
              theme_minimal(base_size = 13) +
              scale_x_continuous(labels = scales::dollar) +
              labs(title = "Transaction amount distribution", x = "Total Amount"))
    }, error = function(e) {
      plot.new(); text(.5,.5, paste("⚠", conditionMessage(e)), cex=1.2, col="grey40")
    })
  })
  
  output$pj_month <- renderPlot({
    tryCatch({
      m <- eco_r() %>% mutate(Month = format(TransactionDate, "%Y-%m")) %>%
        group_by(Month) %>% summarise(Rev = sum(TotalAmount)) %>% arrange(Month)
      print(ggplot(m, aes(Month, Rev, group = 1)) +
              geom_line(color = "darkgreen", linewidth = 1) +
              geom_point(color = "red", size = 2.6) +
              theme_minimal(base_size = 12) +
              theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
              scale_y_continuous(labels = scales::dollar) +
              labs(title = "Monthly revenue trend", y = "Revenue"))
    }, error = function(e) {
      plot.new(); text(.5,.5, paste("⚠", conditionMessage(e)), cex=1.2, col="grey40")
    })
  })
  
  output$pj_mem <- renderPlot({
    tryCatch({
      print(ggplot(eco_r(), aes(MembershipStatus, TotalAmount,
                                fill = MembershipStatus)) +
              geom_boxplot(alpha = .75, outlier.color = "red") +
              theme_minimal(base_size = 13) + theme(legend.position = "none") +
              scale_y_continuous(labels = scales::dollar) +
              labs(title = "Spend by membership tier", x = NULL))
    }, error = function(e) {
      plot.new(); text(.5,.5, paste("⚠", conditionMessage(e)), cex=1.2, col="grey40")
    })
  })
  
  output$pj_sp <- renderPlot({
    tryCatch({
      print(ggplot(eco_r(), aes(Price, TotalAmount, color = ProductCategory)) +
              geom_point(alpha = .65, size = 2) +
              geom_smooth(method = "lm", se = FALSE, color = "grey30", linetype = "dashed") +
              theme_minimal(base_size = 13) +
              scale_x_continuous(labels = scales::dollar) +
              scale_y_continuous(labels = scales::dollar) +
              labs(title = "Unit price vs total amount"))
    }, error = function(e) {
      plot.new(); text(.5,.5, paste("⚠", conditionMessage(e)), cex=1.2, col="grey40")
    })
  })
  
  output$pj_tab <- renderDT({
    tryCatch({
      df <- eco_r() %>% group_by(ProductCategory) %>%
        summarise(Revenue = round(sum(TotalAmount), 2),
                  Transactions = n(),
                  Avg_Value = round(mean(TotalAmount), 2)) %>%
        arrange(desc(Revenue))
      datatable(df, options = list(dom = "t"), rownames = FALSE)
    }, error = function(e)
      datatable(data.frame(Message = paste("⚠", conditionMessage(e)))))
  })
}

shinyApp(ui, server)
