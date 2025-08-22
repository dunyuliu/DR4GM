program read_surfdata
implicit none
!-----------------------------------
integer :: nxt=900 !!number of points fault-prallel
integer :: nyt=250 !! number of points fault-normal
real :: dh=0.1 !!grid sampling in km
integer :: np=1600 !!number of points in time series
real :: dt=0.025 !temporal sampling in s
!-------------------------------------
real, allocatable :: staN(:,:),staE(:,:) !! coordintes of gri points N is FP, E is FN
real, allocatable :: seissurfU(:,:,:),seissurfV(:,:,:),seissurfW(:,:,:) !! velociity time series for surface grid for NEZ, resp.
integer :: i,j,k

    allocate(staN(nxt,nyt),staE(nxt,nyt))
    do j=1,nyt
      do i=1,nxt
        staN(i,j)=dh*i
        staE(i,j)=-dh*(nyt-j+1)
      enddo
    enddo
    
    allocate(seissurfU(np,nxt,nyt),seissurfV(np,nxt,nyt),seissurfW(np,nxt,nyt))
    seissurfU=0.;seissurfV=0.;seissurfW=0.  !Order: NEZ
    open(32,file='seisoutU.surface.gnuplot.dat')
    open(33,file='seisoutV.surface.gnuplot.dat')
    open(34,file='seisoutW.surface.gnuplot.dat')
!    write(*,*)'Reading seismograms...'

    do k=1,np
      do i=1,nxt
        read(32,*)(seissurfU(k,i,j),j=1,nyt)
        read(33,*)(seissurfV(k,i,j),j=1,nyt)
    !    read(34,*)(seissurfW(k,i,j),j=1,nyt) !z component not analyzed yet
      enddo
    enddo
    close(32);close(33);close(34)
    
end program
