# set configure file
configure_file(cmz_ed/CMZedConfig.h.in cmz_ed/CMZedConfig.h)
# install includes
list(APPEND SOURCES_HXX
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
install(FILES ${SOURCES_HXX} DESTINATION include/cmz_ed)
