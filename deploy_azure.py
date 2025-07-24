#!/usr/bin/env python3
"""
Azure deployment script for Health Monitoring System
"""

import os
import subprocess
import json

def deploy_to_azure():
    """Deploy the health monitoring system to Azure"""
    
    print("🚀 Starting Azure deployment...")
    
    # Configuration
    resource_group = "health-monitor-rg"
    app_name = "health-monitor-app"
    location = "eastus"
    
    try:
        # 1. Create resource group
        print("📦 Creating resource group...")
        subprocess.run([
            "az", "group", "create",
            "--name", resource_group,
            "--location", location
        ], check=True)
        
        # 2. Create App Service plan
        print("🏗️  Creating App Service plan...")
        subprocess.run([
            "az", "appservice", "plan", "create",
            "--name", f"{app_name}-plan",
            "--resource-group", resource_group,
            "--sku", "B1",
            "--is-linux"
        ], check=True)
        
        # 3. Create web app
        print("🌐 Creating web app...")
        subprocess.run([
            "az", "webapp", "create",
            "--resource-group", resource_group,
            "--plan", f"{app_name}-plan",
            "--name", app_name,
            "--runtime", "PYTHON|3.9"
        ], check=True)
        
        # 4. Deploy code
        print("📤 Deploying code...")
        subprocess.run([
            "az", "webapp", "deployment", "source", "config-zip",
            "--resource-group", resource_group,
            "--name", app_name,
            "--src", "health_monitor.zip"
        ], check=True)
        
        print("✅ Deployment completed successfully!")
        print(f"🔗 Your app is available at: https://{app_name}.azurewebsites.net")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Deployment failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    deploy_to_azure()
