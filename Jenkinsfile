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
                                sh "cp helloworld.war /etc/apache-tomcat-8.5.61/webapps/"
                        }
                }
	}
}
