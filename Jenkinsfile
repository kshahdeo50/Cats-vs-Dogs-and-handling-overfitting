pipeline {
	    agent any
	    tools {
	        maven 'maven'
	        }
	    stages {
	        stage ('Compile') {
	            steps {
	                sh 'mvn compile'
	                }
	            }      
	        stage ('Package') {
	            steps {
	                sh 'mvn package'
	                }
	            }
	        stage ('Install') {
	            steps {
	                sh 'mvn install'
	                }
	            }
                stage ('Deploy War File') {
                        steps {
                                sh "cp sample.war /root/apache-tomcat-9.0.41/webapps/"
                        }
                }
	}
}