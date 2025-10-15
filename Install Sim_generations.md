Install Sim_generations on Windows

Step 1: **Install Python** (if not already installed)

1\. Go to <https://www.python.org/downloads/windows>

Click on the Downloads button then the Python button under Downloads for
Windows.

Save the python\*\*\*.exe file in your Downloads directory, then
double-click on it.

Be sure to click on the check box "**Add python.exe to PATH**". Very
important!

2\. Click "Install Now".

3\. Click the Close button.

Step 2: **Install Git for Windows** (if not already done in the past)

https://git-scm.com/downloads

Click on \"Windows\"

Click on \"Click here to download\"

Save the Git-\*.exe file in your download directory and then double
click on it.

Accept the default installation directory and all the default
recommendations.

In the cmd window type cd C:\\Users\\\[YourUsername\]\\ (where
Sim_generations is to be installed)

Type: git clone https://github.com/RoyalTr/Sim_generations.git

Step 3: **Set Up a Virtual Environment**

1\. In the cmd window type: cd
C:\\Users\\\[YourUsername\]\\Sim_generations

2\. Type: python -m venv venv

3\. Type: venv\\Scripts\\activate

4\. Type: pip install -r requirements.txt

Each time you want to use Sim_generation you must activate the virtual
environment first:

cmd then type cd C:\\Users\\\[YourUsername\]\\Sim_generations\\

venv\\Scripts\\activate

Run the program you want e.g.,

cd Fix_1_haplo

python Fix_1_haplo.py

To close the venv when done, type exit.