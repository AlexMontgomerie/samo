# get the path
set prj_path [ lindex $argv 2 ]

# run
open_project ${prj_path}
export_design -flow impl -rtl verilog -format ip_catalog
exit

