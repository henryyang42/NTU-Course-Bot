/* ======================== Data Loading ======================== */
libname localLib '/home/NTUCB/data';
libname myCasLib cas caslib=casuser;

PROC CASUTIL ;
    list;
RUN;

%if not (%sysfunc(exist(myCaslib.NTUCourse_1)) AND %sysfunc(exist(myCaslib.NTUCourse_2)) AND 
    %sysfunc(exist(myCaslib.stopList_chinese))) %then %do;

PROC CASUTIL ;
    load file='/home/NTUCB/data/NTUCourse-comment_1.csv' 
        casout="NTUCourse_1"
        replace;
    load file='/home/NTUCB/data/NTUCourse-comment_2.csv' 
        casout="NTUCourse_2"
        replace;
    load data=localLib.stopList_chinese casout="stopList_chinese" replace;
RUN;

%end;

PROC CASUTIL ;
    contents casdata="NTUCourse_1";
    contents casdata="NTUCourse_2";
    contents casdata="stopList_chinese";
RUN;


/* ======================== Sentiment Analysis (part-1) ======================== */
PROC CAS ;
    action sentimentAnalysis.applySent;
    param
        docId="doc_id"
        text="Content"
        language="Chinese"
        table={name="NTUCOURSE_1"}
        casOut={name="NTUCOURSE_1_sentiment" , replace=TRUE};
RUN;

/* ======================== Sentiment Analysis (part-2) ======================== */
PROC CAS ;
    action sentimentAnalysis.applySent;
    param
        docId="doc_id"
        text="Content"
        language="Chinese"
        table={name="NTUCOURSE_2"}
        casOut={name="NTUCOURSE_2_sentiment" , replace=TRUE};
RUN;

/* ======================== Data Merging (part-1) ======================== */
Data localLib.NTUCOURSE_1;
    merge myCasLib.NTUCOURSE_1_sentiment myCasLib.NTUCOURSE_1;
    by doc_id;
    sentiment = _sentiment_;
    probability = _probability_;
    drop _sentiment_ _probability_;
RUN;

/* ======================== Data Merging (part-2) ======================== */
Data localLib.NTUCOURSE_2;
    merge myCasLib.NTUCOURSE_2_sentiment myCasLib.NTUCOURSE_2;
    by doc_id;
    sentiment = _sentiment_;
    probability = _probability_;
    drop _sentiment_ _probability_;
RUN;

/* ======================== Data Exporting (part-1) ======================== */
PROC EXPORT data=locallib.ntucourse_1
    OUTFILE='/home/NTUCB/data/NTUCourse-comment_1_sentiment.csv'
    DBMS=CSV REPLACE;
    DELIMITER=',';
RUN;

/* ======================== Data Exporting (part-2) ======================== */
PROC EXPORT data=locallib.ntucourse_2
    OUTFILE='/home/NTUCB/data/NTUCourse-comment_2_sentiment.csv'
    DBMS=CSV REPLACE;
    DELIMITER=',';
RUN;
