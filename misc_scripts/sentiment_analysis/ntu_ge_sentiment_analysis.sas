/* ======================== Data Loading ======================== */
libname localLib '/home/NTUCB/data';
libname myCasLib cas caslib=casuser;

PROC CASUTIL ;
    list;
RUN;

%if not (%sysfunc(exist(myCaslib.NTUCourse_GE)) AND 
    %sysfunc(exist(myCaslib.stopList_chinese))) %then %do;

PROC CASUTIL ;
    load file='/home/NTUCB/data/NTUCourse-comment_ge.csv' 
        casout="NTUCourse_GE"
        replace;
    load data=localLib.stopList_chinese casout="stopList_chinese" replace;
RUN;

%end;

PROC CASUTIL ;
    contents casdata="NTUCourse_GE";
    contents casdata="stopList_chinese";
RUN;


/* ======================== Sentiment Analysis ======================== */
PROC CAS ;
    action sentimentAnalysis.applySent;
    param
        docId="doc_id"
        text="Content"
        language="Chinese"
        table={name="NTUCourse_GE"}
        casOut={name="NTUCourse_GE_sentiment" , replace=TRUE};
RUN;


/* ======================== Data Merging ======================== */
Data localLib.NTUCourse_GE;
    merge myCasLib.NTUCourse_GE_sentiment myCasLib.NTUCourse_GE;
    by doc_id;
    sentiment = _sentiment_;
    probability = _probability_;
    drop _sentiment_ _probability_;
RUN;


/* ======================== Data Exporting ======================== */
PROC EXPORT data=locallib.NTUCourse_GE
    OUTFILE='/home/NTUCB/data/NTUCourse-comment_1_sentiment.csv'
    DBMS=CSV REPLACE;
    DELIMITER=',';
RUN;
