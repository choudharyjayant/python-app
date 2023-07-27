def sendEmail(Status){
    emailext subject: "CI CD Update ||Dev Question Generator API || Job #${BUILD_NUMBER} || Status ${Status}",
                    body: "Project : ${JOB_NAME}<br>Build Status: ${Status}<br>Build URL: $BUILD_URL<br>Build Duration : ${currentBuild.durationString.replace(' and counting', '')}<br> Build Date : ${new Date().format('dd/MM/yyyy')}",mimeType: 'text/html',
                    replyTo: '$DEFAULT_REPLYTO',
                    to: "tushar.khachane@magicedtech.com" ,
                    attachLog: true
}
def status

node('AI_ML_Practice_Windows_Slave') {
    properties([
    parameters([
        choice(choices: ['jenkins_windows_slave'], description: "Please Select env", name: "Branch"),
        string(name: "BUCKET", defaultValue: "cloudcoe-abp-wars", description: "bucketName"),
        string(name: "REGION", defaultValue: "ap-south-1", description: "regionName")
    ])
    ])
    try {
        stage('Clone Repository') { 
            cleanWs()
            git url: 'https://gitlab.magicsw.com/accelerator/question-generator/question-generator-api.git', branch: params.Branch , credentialsId: 'TusharGitlabKIPS'
        }
        
        stage('Deploy') {
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: 'CloudCoe_ABP_ECR']]) {
                    def JOB_BASE_NAME = "${env.JOB_NAME}".split('/').last()
                    def destinationFile = "${env.JOB_BASE_NAME}/${params.Branch}/${env.JOB_BASE_NAME}.zip"
                    def versionLabel = "${env.JOB_BASE_NAME}#${params.Branch}#${env.BUILD_NUMBER}"
                    def description = "${env.BUILD_URL}"
                    bat """
                        dir
                        xcopy * "D:/AI_ML_Practice/test/question-generator-api" /E/C/I/Y
                        pushd "D:/AI_ML_Practice/test/question-generator-api"
                        dir
                        popd
                    """
                    //start /B python main.py
                    sh 'nohup python /d/AI_ML_Practice/test/question-generator-api/main.py > question-generator-api.log &'
                    
                }
        }
        
        stage('post ') {
                status = "SUCCESS"
                //cleanWs deleteDirs: true, disableDeferredWipeout: true, patterns: [[pattern: '**/build/**', type: 'EXCLUDE']]
                
        }
    }
     catch(Exception e){
        status = "FAILURE"
         //sendEmail(status)
         //cleanWs deleteDirs: true, disableDeferredWipeout: true, patterns: [[pattern: '**/build/**', type: 'EXCLUDE']]
        throw e;
    }
    finally{
       echo "Done"
    }
}
