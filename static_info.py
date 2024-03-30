db_schema="""
CREATE TABLE patients 
(
    row_id INT NOT NULL PRIMARY KEY, -- Unique ID of the patient
    subject_id INT NOT NULL UNIQUE, -- Unique subject id of the patient
    gender VARCHAR(5) NOT NULL, -- Gender of the patient
    dob TIMESTAMP(0) NOT NULL, -- Date of birth of the patient
    dod TIMESTAMP(0) -- Date of death of the patient
);
CREATE TABLE admissions
(
    row_id INT NOT NULL PRIMARY KEY, -- Unique ID of the admission
    subject_id INT NOT NULL, -- Subject id of the admission
    hadm_id INT NOT NULL UNIQUE, -- Unique hospital admission id of the admission
    admittime TIMESTAMP(0) NOT NULL, -- Admit time of the admission
    dischtime TIMESTAMP(0), -- Discharge time of the admission
    admission_type VARCHAR(50) NOT NULL, -- Admission type of the admission
    admission_location VARCHAR(50) NOT NULL, -- Admission location of the admission
    discharge_location VARCHAR(50), -- Discharge location of the admission
    insurance VARCHAR(255) NOT NULL, -- Insurance of the admission
    language VARCHAR(10), -- Langauge of the admission
    marital_status VARCHAR(50), -- Marital status of the admission
    age INT NOT NULL, -- Age of the admission
);
CREATE TABLE d_icd_diagnoses
(
    row_id INT NOT NULL PRIMARY KEY, -- Unique ID of the icd diagnose
    icd_code VARCHAR(10) NOT NULL UNIQUE, -- Unique icd code of the icd diagnose
    long_title VARCHAR(255) NOT NULL -- Title of the icd diagnose
);
CREATE TABLE d_icd_procedures 
(
    row_id INT NOT NULL PRIMARY KEY, -- Unique ID of icd procedure
    icd_code VARCHAR(10) NOT NULL UNIQUE, -- Unique icd code of the icd procedure
    long_title VARCHAR(255) NOT NULL -- Title of the icd procedure
);
CREATE TABLE d_labitems 
(
    row_id INT NOT NULL PRIMARY KEY, -- Unique ID of the item relate to laboratory tests
    itemid INT NOT NULL UNIQUE, -- Unique item id of the item relate to laboratory tests
    label VARCHAR(200) -- Label of the item relate to laboratory tests
);
CREATE TABLE d_items 
(
    row_id INT NOT NULL PRIMARY KEY, -- Unique ID of the item excepts item relate to laboratory tests
    itemid INT NOT NULL UNIQUE, -- Unique item id of the item excepts item relate to laboratory tests
    label VARCHAR(200) NOT NULL, -- Label of item excepts item relate to laboratory tests
    abbreviation VARCHAR(200) NOT NULL, -- Abbreviation of item excepts item relate to laboratory tests
    linksto VARCHAR(50) NOT NULL -- Event linked to item excepts item relate to laboratory tests
);
CREATE TABLE diagnoses_icd
(
    row_id INT NOT NULL PRIMARY KEY, -- Unique ID of diagnose
    subject_id INT NOT NULL, -- Subject id of diagnose
    hadm_id INT NOT NULL, -- Hospital admission id of diagnose
    icd_code VARCHAR(10) NOT NULL, -- ICD code of diagnose
    charttime TIMESTAMP(0) NOT NULL, -- Chart time of diagnose
);
CREATE TABLE procedures_icd
(
    row_id INT NOT NULL PRIMARY KEY, -- Unique ID of procedures
    subject_id INT NOT NULL, -- Subject id of procedures
    hadm_id INT NOT NULL, -- Hospital admission id of procedures
    icd_code VARCHAR(10) NOT NULL, -- ICD code of procedures
    charttime TIMESTAMP(0) NOT NULL, -- Chart time of procedures
);
CREATE TABLE labevents
(
    row_id INT NOT NULL PRIMARY KEY, -- Unique ID of laboratory event
    subject_id INT NOT NULL, -- Subject id of laboratory event
    hadm_id INT NOT NULL, -- Hospital admission id of laboratory event
    itemid INT NOT NULL, -- Item id of laboratory event
    charttime TIMESTAMP(0), -- Chart time of laboratory event
    valuenum DOUBLE PRECISION, -- Numerical value measured of laboratory event
    valueuom VARCHAR(20), -- Unit of numerical value of laboratory event
);
CREATE TABLE prescriptions
(
    row_id INT NOT NULL PRIMARY KEY, -- Unique ID of prescription
    subject_id INT NOT NULL, -- Subject id of prescription
    hadm_id INT NOT NULL, -- Hospital admission id of prescription
    starttime TIMESTAMP(0) NOT NULL, -- Start time of prescription
    stoptime TIMESTAMP(0), -- Stop time of prescription
    drug VARCHAR(255) NOT NULL, -- Drug name of prescription
    dose_val_rx VARCHAR(100) NOT NULL, -- Dosage value of prescription
    dose_unit_rx VARCHAR(50) NOT NULL, -- Dosage unit of prescription
    route VARCHAR(50) NOT NULL, -- Intake method of prescription
);
CREATE TABLE cost
(
    row_id INT NOT NULL PRIMARY KEY, -- Unique ID of cost event
    subject_id INT NOT NULL, -- Subject id of cost event
    hadm_id INT NOT NULL, -- Hospital admission id of cost event
    event_type VARCHAR(20) NOT NULL, -- Event type of cost event
    event_id INT NOT NULL, -- Event id of cost event
    chargetime TIMESTAMP(0) NOT NULL, -- Charge time of cost event
    cost DOUBLE PRECISION NOT NULL, -- Cost of cost event
);
CREATE TABLE chartevents
(
    row_id INT NOT NULL PRIMARY KEY, -- Unique ID of chart event
    subject_id INT NOT NULL, -- Subject id of chart event
    hadm_id INT NOT NULL, -- Hospital admission id of chart event
    stay_id INT NOT NULL, -- Stay ID of chart event
    itemid INT NOT NULL, -- Item ID of chart event
    charttime TIMESTAMP(0) NOT NULL, -- Chart time of chart event
    valuenum DOUBLE PRECISION, -- Numerical value measured of chart event
    valueuom VARCHAR(50), -- Unit of numerical value of chart event
);
CREATE TABLE inputevents
(
    row_id INT NOT NULL PRIMARY KEY, -- Unique ID of input event
    subject_id INT NOT NULL, -- Subject id of input event
    hadm_id INT NOT NULL, -- Hospital admission id of input event
    stay_id INT NOT NULL, -- Stay id of input event
    starttime TIMESTAMP(0) NOT NULL, -- Start time of input event
    itemid INT NOT NULL, -- Item id of input event
    amount DOUBLE PRECISION, -- Amount of input event
);
CREATE TABLE outputevents
(
    row_id INT NOT NULL PRIMARY KEY, -- Unique ID of output event
    subject_id INT NOT NULL, -- Subject id of output event
    hadm_id INT NOT NULL, -- Hospital admission id of output event
    stay_id INT NOT NULL, -- Stay id of output event
    charttime TIMESTAMP(0) NOT NULL, -- Chart time of output event
    itemid INT NOT NULL, -- Item id of output event
    value DOUBLE PRECISION, -- Value of output event
);
CREATE TABLE microbiologyevents
(
    row_id INT NOT NULL PRIMARY KEY, -- Unique ID of microbiologyevent
    subject_id INT NOT NULL, -- Subject id of microbiologyevent
    hadm_id INT NOT NULL, -- Hospital admission id of microbiologyevent
    charttime TIMESTAMP(0) NOT NULL, -- Chart time of microbiologyevent
    spec_type_desc VARCHAR(100), -- Specimen name of microbiologyevent
    test_name VARCHAR(100), -- Test name of microbiologyevent
    org_name VARCHAR(100), -- Organism name of microbiologyevent
);
CREATE TABLE icustays
(
    row_id INT NOT NULL PRIMARY KEY, -- Unique ID of icu stay
    subject_id INT NOT NULL, -- Subject id of icu stay
    hadm_id INT NOT NULL, -- Hospital admission id of icu stay
    stay_id INT NOT NULL UNIQUE, -- Stay id of icu stay
    first_careunit VARCHAR(20) NOT NULL, -- first care unit of icu stay
    last_careunit VARCHAR(20) NOT NULL, -- Last care unit of icu stay
    intime TIMESTAMP(0) NOT NULL, -- In time of icu stay
    outtime TIMESTAMP(0), -- Out time of icu stay
);
CREATE TABLE transfers
(
    row_id INT NOT NULL PRIMARY KEY, -- Unique ID of transfer
    subject_id INT NOT NULL, -- Subject Id of transfer
    hadm_id INT NOT NULL, -- Hospital admission id of transfer
    transfer_id INT NOT NULL, -- Transfer Id of transfer
    eventtype VARCHAR(20) NOT NULL, -- Event type of transfer
    careunit VARCHAR(20), -- Care unit of transfer
    intime TIMESTAMP(0) NOT NULL, -- In time of transfer
    outtime TIMESTAMP(0), -- Out time of transfer
);
"""
db_table_information="""
ADMISSIONS -- Every unique hospitalization for each patient in the database
PATIENTS -- Every unique patient in the database
D_ICD_DIAGNOSES -- International Statistical Classification of Diseases and Related Health Problems (ICD-9) codes relating to diagnoses
D_ICD_PROCEDURES -- International Statistical Classification of Diseases and Related Health Problems (ICD-9) codes relating to procedures
D_LABITEMS -- Local codes ('ITEMIDs') appearing in the database that relate to laboratory tests
D_ITEMS -- Local codes ('ITEMIDs') appearing in the database, except those that relate to laboratory tests
DIAGNOSES_ICD -- Hospital assigned diagnoses, coded using the International Statistical Classification of Diseases and Related Health Problems (ICD) system
PROCEDURES_ICD -- Patient procedures, coded using the International Statistical Classification of Diseases and Related Health Problems (ICD) system
LABEVENTS -- Laboratory measurements for patients both within the hospital and in outpatient clinics
PRESCRIPTIONS -- Medications ordered for a given patient
COST -- All patients events cost
CHARTEVENTS -- All charted observations for patients
INPUTEVENTS -- Intake for patients monitored while in the ICU
OUTPUTEVENTS -- Output information for patients while in the ICU
MICROBIOLOGYEVENTS -- Microbiology culture results and antibiotic sensitivities from the hospital database
ICUSTAYS -- Every unique ICU stay in the database
TRANSFERS -- Patient movement from bed to bed within the hospital
"""