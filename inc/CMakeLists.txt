# set configure file
configure_file(cmz_ed/CMZedConfig.h.in cmz_ed/CMZedConfig.h)
# install includes
list(APPEND ASCI_SOURCES_HXX
     ${PROJECT_BINARY_DIR}/inc/cmz_ed/CMZedConfig.h
     cmz_ed/eigsolver.h++
     cmz_ed/integrals.h++
     cmz_ed/lanczos.h++
     cmz_ed/rdms.h++
     cmz_ed/combins.h++
     cmz_ed/slaterdet.h++
     cmz_ed/hamil.h++
     cmz_ed/ed.h++
     cmz_ed/utils.h++
    )
list(APPEND DBWY_SOURCES_HXX
     dbwy/asci_body.hpp
     dbwy/asci_util.hpp
     dbwy/bitset_operations.hpp
     dbwy/cmz_compat_csr_matrix.hpp
     dbwy/csr_hamiltonian.hpp
     dbwy/davidson.hpp
     dbwy/double_loop.hpp
     dbwy/hamiltonian_generator.hpp
     dbwy/residue_arrays.hpp
     dbwy/sd_operations.hpp
    )
install(FILES ${ASCI_SOURCES_HXX} DESTINATION include/cmz_ed)
install(FILES ${DBWY_SOURCES_HXX} DESTINATION include/dbwy)
