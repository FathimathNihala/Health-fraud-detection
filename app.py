from flask import Flask,request,render_template
import pickle
import numpy as np

with open('model.pkl','rb') as model_file:
    model=pickle.load(model_file)

app=Flask(__name__)
                      
@app.route('/')
def home():
   return render_template('index.html')

@app.route('/detection',methods=['POST'])
def detect():
   BeneID=request.form['Beneficiary ID']
   BeneID=int(BeneID.replace('BENE',''))
   DOB=request.form['Date Of Birth']
   BirthYear=int(DOB[0:4])
   ChronicCond_Alzheimer=int(request.form['Alzheimer'])
   ChronicCond_Heartfailure=int(request.form['Heart Failure'])
   ChronicCond_KidneyDisease=int(request.form['Chronic Kidney Disease'])
   ChronicCond_Cancer=int(request.form['Cancer'])
   ChronicCond_ObstrPulmonary=int(request.form['Obstructive Pulmonary'])
   ChronicCond_Depression=int(request.form['Depression'])
   ChronicCond_Diabetes=int(request.form['Diabetes'])
   ChronicCond_IschemicHeart=int(request.form['Ischemic Heart'])
   ChronicCond_Osteoporasis=int(request.form['Osteo Porosis'])
   ChronicCond_rheumatoidarthritis=int(request.form['Rheumatoid Arthritis'])
   ChronicCond_stroke=int(request.form['Stroke'])
   InscClaimAmtReimbursed=int(request.form['Insurance Claim Amount Reimbursed'])
   IPAnnualReimbursementAmt=int(request.form['IP Annual Reimbursement Amount'])
   OPAnnualReimbursementAmt=int(request.form['OP Annual Reimbursement Amount'])
   IPAnnualDeductibleAmt=int(request.form['IP Annual Deductible Amount'])
   OPAnnualDeductibleAmt=int(request.form['OP Annual Deductible Amount'])
   TotalAnnualReimbursementAmt=IPAnnualReimbursementAmt+OPAnnualReimbursementAmt
   TotalAnnualDeductibleAmt=IPAnnualDeductibleAmt+OPAnnualDeductibleAmt
   Provider=request.form['Provider ID']
   Provider=int(Provider.replace('PRV',''))
   AttendingPhysician=request.form['Attending Physician ID']
   AttendingPhysician=int(AttendingPhysician.replace('PHY',''))
   OperatingPhysician=request.form['Operating Physician ID']
   OperatingPhysician=int(OperatingPhysician.replace('PHY',''))
   OtherPhysician=request.form['Other Physician ID']
   OtherPhysician=int(OtherPhysician.replace('PHY',''))
   ClmAdmitDiagnosisCode=request.form['Claim Admit Diagnosis Code']
   ClmAdmitDiagnosisCode=ClmAdmitDiagnosisCode.replace('V','')
   ClmAdmitDiagnosisCode=ClmAdmitDiagnosisCode.replace('E','')
   ClmAdmitDiagnosisCode=int(ClmAdmitDiagnosisCode)
   feature=np.array([[BeneID,BirthYear,ChronicCond_Alzheimer,ChronicCond_Heartfailure,ChronicCond_KidneyDisease,
                      ChronicCond_Cancer,ChronicCond_ObstrPulmonary,ChronicCond_Depression,ChronicCond_Diabetes,
                      ChronicCond_IschemicHeart,ChronicCond_Osteoporasis,ChronicCond_rheumatoidarthritis,ChronicCond_stroke,
                      InscClaimAmtReimbursed,TotalAnnualReimbursementAmt,TotalAnnualDeductibleAmt,Provider,AttendingPhysician,
                      OperatingPhysician,OtherPhysician,ClmAdmitDiagnosisCode]])
   prediction=model.predict(feature)
   return render_template('result.html',pred_res=prediction[0])

if __name__=='__main__':
  app.run(debug=True)