# ZEBRA software

This is a compilation of software created for the ZEBRA instrument at PSI. This software comes from a list of old software implemented in Fortran, but is now implemented in Nicos. The list of current software is:
- PyRefine
-   A UB Matrix Refinement algorithm for Normal Beam, Eulerian Cradle and Kappa Geometries
-   Currently works perfectly for NB
-   Needs to fix the calculations of omega in Eulerian Cradle
-   Kappa to be fixed once that is relevant
- XYZ
-   Calculation from two Friedel Reflection Pairs to figure out displacement of sample

A lot of different software is also on the way:
- Incommensorate reflection: Calculate propogation vector
- Optimal measurement path
- Figuring out resolution of Zebra out of plane
-   Calculations of this
- Optimizing measurements for step size depending on how out of plane NB is
- Kappa geometry for neutron diffraction

This project is done in collaboration with Romain Sibille at PSI. For any questions or requests, please write to nicolai.amin@psi.ch
